import time
import os
import sys
import argparse
import pprint
import pickle

import numpy as np
import torch

from create import (
    create_envs,
    create_threads,
    SelfplayActGroup,
)
from eval import evaluate
import common_utils
import rela
import hanalearn  # type: ignore
import r2d2
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--config", type=str, default=None)

    # training setup related
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--save_per", type=int, default=50)
    parser.add_argument("--load_model", type=str, default="None")
    parser.add_argument("--seed", type=str, default="a")
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--act_device", type=str, default="cuda:0")
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--target_data_ratio", type=float, default=None, help="train/gen")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # algo setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--num_eps", type=float, default=80)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--multi_step", type=int, default=1)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--vdn", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)

    # optim setting
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=128)

    # model setting
    parser.add_argument("--net", type=str, default="publ-lstm", help="publ-lstm/lstm/ffwd")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    # replay/data settings
    parser.add_argument("--replay_buffer_size", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")
    parser.add_argument("--burn_in_frames", type=int, default=1000)

    # llm setting
    parser.add_argument("--llm_prior", type=str, default=None)
    parser.add_argument("--pikl_lambda", type=float, default=0.0)
    parser.add_argument("--pikl_anneal_per", type=int, default=10)
    parser.add_argument("--pikl_anneal_rate", type=float, default=0.5)
    parser.add_argument("--pikl_beta", type=float, default=1)
    parser.add_argument("--llm_noise", type=float, default=0.0)
    parser.add_argument("--llm_noise_seed", type=int, default=97)

    # game config
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)
    parser.add_argument("--num_color", type=int, default=5)
    parser.add_argument("--num_rank", type=int, default=5)
    parser.add_argument("--num_hint", type=int, default=8)
    parser.add_argument("--bomb", type=int, default=0)

    # debug
    parser.add_argument("--do_eval", type=int, default=1)

    args = parser.parse_args()
    args = common_utils.maybe_load_config(args)

    args.seed = utils.get_seed(args.seed)
    return args


def train(args):
    common_utils.set_all_seeds(args.seed)

    logger_path = os.path.join(args.save_dir, f"train.log")
    sys.stdout = common_utils.Logger(logger_path, print_to_stdout=True)
    pprint.pprint(vars(args))
    utils.print_gpu_info()

    train_device = args.train_device
    act_device = args.act_device

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.bomb,
        args.max_len,
        hand_size=args.hand_size,
        num_color=args.num_color,
        num_rank=args.num_rank,
        num_hint=args.num_hint,
    )

    agent = r2d2.R2D2Agent(
        args.vdn,
        args.multi_step,
        args.gamma,
        train_device,
        games[0].feature_size(False),
        args.rnn_hid_dim,
        games[0].num_action(),
        args.net,
        args.num_lstm_layer,
        off_belief=False,
    )
    print(agent)

    if args.load_model and args.load_model != "None":
        print("*****loading pretrained model*****")
        print(args.load_model)
        online_net = utils.get_agent_online_network(agent, False)
        utils.load_weight(online_net, args.load_model, train_device)
        print("***************done***************")

    saver = common_utils.TopkSaver(args.save_dir, 5)
    online_net = utils.get_agent_online_network(agent, False)
    optim = torch.optim.Adam(online_net.parameters(), lr=args.lr, eps=args.eps)

    replay_buffer = rela.RNNReplay(  # type: ignore
        args.replay_buffer_size,
        args.seed,
        args.prefetch,
    )

    explore_eps = utils.generate_explore_eps(args.act_base_eps, args.act_eps_alpha, args.num_eps)
    eps_str = [[f"\n{eps:.9f}", f"{eps:.9f}"][i % 5 != 0]  for i, eps in enumerate(explore_eps)]
    print("explore eps:", ", ".join(eps_str))
    print("avg explore eps:", np.mean(explore_eps))

    # making input arguments
    act_group_args = {
        "devices": act_device,
        "agent": agent,
        "seed": args.seed,
        "num_thread": args.num_thread,
        "num_game_per_thread": args.num_game_per_thread,
        "num_player": args.num_player,
        "replay_buffer": replay_buffer,
        "method_batchsize": {"act": 5000},
        "explore_eps": explore_eps,
    }

    act_group_args["actor_args"] = {
        "num_player": args.num_player,
        "vdn": args.vdn,
        "sad": False,
        "shuffle_color": args.shuffle_color,
        "hide_action": False,
        "trinary": hanalearn.AuxType.Trinary,
        "multi_step": args.multi_step,
        "seq_len": args.max_len,
        "gamma": args.gamma,
    }

    llm_prior = None
    if args.llm_prior is not None:
        llm_prior = utils.load_and_process_llm_prior(args.llm_prior, games[0], verbose=False)
        if args.llm_noise > 0:
            flipped = 0
            total = 0
            rng =  np.random.default_rng(seed=args.llm_noise_seed)
            for key, vals in llm_prior.items():
                for i, v in enumerate(vals):
                    total += 1
                    if rng.random() < args.llm_noise:
                        assert llm_prior[key][i] == 0 or llm_prior[key][i] == 1
                        llm_prior[key][i] = 1 - v
                        flipped += 1
            print(f"total #flipped: {flipped}, percentage: {100 * flipped/total:.1f}%")

        pkl_path = os.path.join(args.save_dir, "llm.pkl")
        print(f"dumping llm_prior to {pkl_path}")
        pickle.dump(llm_prior, open(pkl_path, "wb"))
        act_group_args["llm_prior"] = llm_prior
        act_group_args["pikl_beta"] = args.pikl_beta
        act_group_args["pikl_lambda"] = [args.pikl_lambda]

    act_group_type = SelfplayActGroup
    act_group = act_group_type(**act_group_args)
    context, _ = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )

    act_group.start()
    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)
    print("Success, Done")
    print("=" * 100)

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    sleep_time = 0

    for epoch in range(args.num_epoch):
        if (args.pikl_lambda > 0
            and epoch > 0
            and args.pikl_anneal_per > 0
            and epoch % args.pikl_anneal_per == 0
        ):
            args.pikl_lambda *= args.pikl_anneal_rate
            for actor in act_group.flat_actors:
                actor.update_llm_lambda([args.pikl_lambda])

        print(f"EPOCH: {epoch}, pikl_lambda={args.pikl_lambda}")
        print(common_utils.get_mem_usage("(epoch start)"))
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            with stopwatch.time("sync & update"):
                num_update = batch_idx + epoch * args.epoch_len
                if num_update % args.num_update_between_sync == 0:
                    agent.sync_target_with_online()
                if num_update % args.actor_sync_freq == 0:
                    act_group.update_model(agent)
                torch.cuda.synchronize()

            with stopwatch.time("sample data"):
                batch = replay_buffer.sample(args.batchsize, train_device)

            with stopwatch.time("forward & backward"):
                loss = agent.loss(batch, args.aux_weight, stat)
                loss = loss.mean()
                loss.backward()
                torch.cuda.synchronize()

            with stopwatch.time("optim step"):
                g_norm = torch.nn.utils.clip_grad_norm_(online_net.parameters(), args.grad_clip)
                optim.step()
                optim.zero_grad()
                torch.cuda.synchronize()

            with stopwatch.time("sleep"):
                if sleep_time > 0:
                    time.sleep(sleep_time)

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        with stopwatch.time("eval & others"):
            count_factor = 1 if not args.vdn else args.num_player
            new_sleep_time = tachometer.lap(
                replay_buffer,
                args.epoch_len * args.batchsize,
                count_factor,
                num_batch=args.epoch_len,
                target_ratio=args.target_data_ratio,
                current_sleep_time=sleep_time,
            )
            sleep_time = 0.6 * sleep_time + 0.4 * new_sleep_time
            print(
                f"Sleep info: new_sleep_time: {int(1000 * new_sleep_time)} MS, "
                f"actual_sleep_time: {int(1000 * sleep_time)} MS"
            )

            stat.summary(epoch)

            context.pause()
            if args.do_eval:
                print(common_utils.get_mem_usage("(before eval)"))
                score, perfect, *_ = evaluate(
                    [agent],
                    1000,
                    np.random.randint(100000),
                    args.bomb,
                    num_player=args.num_player,
                    pikl_lambdas=None if llm_prior is None else [args.pikl_lambda],
                    pikl_betas=None if llm_prior is None else [args.pikl_beta],
                    llm_priors=None if llm_prior is None else [llm_prior],
                    hand_size=args.hand_size,
                    num_color=args.num_color,
                    num_rank=args.num_rank,
                    num_hint=args.num_hint,
                )
                perfect *= 100
                print(common_utils.get_mem_usage("(after eval)"))
            else:
                score, perfect = None, None

            force_save = f"epoch{epoch + 1}" if (epoch + 1) % args.save_per == 0 else None
            model_saved = saver.save(
                online_net.state_dict(), score, force_save_name=force_save, config=vars(args)
            )
            print(
                "Eval(epoch %d): score: %.4f, perfect: %.2f, model saved: %s"
                % (epoch, score, perfect, model_saved)
            )
            context.resume()

        stopwatch.summary()
        print("=" * 100)

    # force quit, "nicely"
    os._exit(0)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # type: ignore
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
