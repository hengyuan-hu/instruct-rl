import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch

from create import *
import common_utils
import rela
import utils
import belief_model


def parse_args():
    parser = argparse.ArgumentParser(description="train belief model")
    parser.add_argument("--save_dir", type=str, default="exps/belief1")
    parser.add_argument("--save_per", type=int, default=50)
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--act_device", type=str, default="cuda:0")

    # load policy config
    parser.add_argument("--policy", type=str, default="")
    parser.add_argument("--explore", type=int, default=1)
    parser.add_argument("--rand", type=int, default=0)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--epoch_len", type=int, default=1000)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=20000)
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=80)

    args = parser.parse_args()
    return args


def create_rl_context(args):
    agent_overwrite = {"device": args.train_device}
    agent, cfgs = utils.load_agent(args.policy, agent_overwrite)

    replay_buffer = rela.RNNReplay(  # type: ignore
        args.replay_buffer_size,
        args.seed,
        args.prefetch,
    )

    if args.rand:
        explore_eps = [1]
    elif args.explore:
        # use the same exploration config as policy learning
        explore_eps = utils.generate_explore_eps(
            cfgs["act_base_eps"], cfgs["act_eps_alpha"], cfgs["num_game_per_thread"]
        )
    else:
        explore_eps = [0]

    eps_str = [[f"\n{eps:.9f}", f"{eps:.9f}"][i % 5 != 0] for i, eps in enumerate(explore_eps)]
    print("explore eps:", ", ".join(eps_str))
    print("avg explore eps:", np.mean(explore_eps))

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        cfgs["num_player"],
        cfgs.get("train_bomb", 0),
        cfgs["max_len"],
    )

    act_group_args = {
        "devices": args.act_device,
        "agent": agent,
        "seed": args.seed,
        "num_thread": args.num_thread,
        "num_game_per_thread": args.num_game_per_thread,
        "num_player": cfgs["num_player"],
        "replay_buffer": replay_buffer,
        "method_batchsize": {"act": 5000},
        "explore_eps": explore_eps,
    }

    act_group_args["actor_args"] = {
        "num_player": cfgs["num_player"],
        "vdn": False,
        "sad": False,
        "shuffle_color": False,
        "hide_action": False,
        "trinary": hanalearn.AuxType.Full,
        "multi_step": cfgs["multi_step"],
        "seq_len": cfgs["max_len"],
        "gamma": cfgs["gamma"],
    }

    act_group = SelfplayActGroup(**act_group_args)

    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )
    return agent, cfgs, replay_buffer, games, act_group, context, threads


if __name__ == "__main__":
    torch.backends.cuda.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 2)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    (
        agent,
        cfgs,
        replay_buffer,
        games,
        act_group,
        context,
        threads,
    ) = create_rl_context(args)
    act_group.start()
    context.start()

    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")

    if len(args.load_model) > 0:
        belief_config = utils.get_train_config(cfgs["belief_model"])
        print("load belief model from:", cfgs["belief_model"])
        model = belief_model.ARBeliefModel.load(
            cfgs["belief_model"],
            args.train_device,
            5,
        )
    else:
        in_dim = games[0].feature_size(cfgs.get("sad", 0))[1]
        model = belief_model.ARBeliefModel(
            args.train_device,
            in_dim,
            args.hid_dim,
            hand_size=5,  # hand_size
            out_dim=25,  # bits per card
        ).to(args.train_device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    sleep_time = 0

    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch}")
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            with stopwatch.time("sample data"):
                batch = replay_buffer.sample(args.batchsize, args.train_device)

            with stopwatch.time("forward & backward"):
                loss, xent, xent_v0, _ = model.loss(batch)
                loss = loss.mean()
                loss.backward()
                torch.cuda.synchronize()

            with stopwatch.time("optim step"):
                g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
                optim.zero_grad()
                torch.cuda.synchronize()

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            stat["xent_pred"].feed(xent.detach().mean().item())
            stat["xent_v0"].feed(xent_v0.detach().mean().item())

            with stopwatch.time("sleep"):
                if sleep_time > 0:
                    time.sleep(sleep_time)

        new_sleep_time = tachometer.lap(
            replay_buffer,
            args.epoch_len * args.batchsize,
            factor=1,
            num_batch=args.epoch_len,
            target_ratio=2,
            current_sleep_time=sleep_time,
        )
        sleep_time = 0.6 * sleep_time + 0.4 * new_sleep_time
        print(
            f"Sleep info: new_sleep_time: {int(1000 * new_sleep_time)} MS, "
            f"actual_sleep_time: {int(1000 * sleep_time)} MS"
        )

        force_save = f"epoch{epoch + 1}" if (epoch + 1) % args.save_per == 0 else None
        saver.save(
            model.state_dict(), -stat["loss"].mean(), save_latest=True, force_save_name=force_save
        )

        stat.summary(epoch)
        stopwatch.summary()
        print("===================")

    # force quit, "nicely"
    os._exit(0)
