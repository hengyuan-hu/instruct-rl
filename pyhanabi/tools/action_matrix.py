import argparse
import os
import sys
import time
import pickle
import numpy as np

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import common_utils
from parse_handshake import *
from parse_log import *
from create import *
import utils
import rela


import matplotlib.pyplot as plt
plt.switch_backend("agg")
plt.rc("image", cmap="viridis")
plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
plt.rc("ytick", labelsize=10)
plt.rc("axes", labelsize=10)
plt.rc("axes", titlesize=10)


def create_dataset_new(
    weight_file,
    *,
    num_game=1000,
    num_thread=10,
    random_start=1,
    device
):
    runners = []
    configs = []

    agent, config = utils.load_agent(
        weight_file,
        {
            "device": device,
            "off_belief": False,
        },
    )
    agent.train(False)
    runner = rela.BatchRunner(agent, device, 500, ["act"])
    runners.append(runner)
    configs.append(config)

    replay_buffer = rela.RNNReplay(
        num_game,  # args.dataset_size,
        1,  # args.seed,
        0,  # args.prefetch,
    )

    game_per_thread = 1
    games = create_envs(
        num_thread * game_per_thread,
        1,  # seed
        configs[0]["num_player"],
        0,  # bomb
        80,  # config["max_len"],
        random_start_player=random_start,
        hand_size=configs[0].get("hand_size", 5),
        num_color=configs[0].get("num_color", 5),
        num_rank=configs[0].get("num_rank", 5),
        num_hint=configs[0].get("num_hint", 8),
    )

    actors = []
    seed = 0
    for i in range(num_thread):
        thread_actor = []
        for player_idx in range(config["num_player"]):
            runner = runners[player_idx % len(runners)]
            config = configs[player_idx % len(runners)]

            seed += 1
            actor = hanalearn.R2D2Actor(
                runner,
                seed,
                config["num_player"],
                player_idx,
                True, # collect data as if we are in vdn
                False, # config["sad"],  # sad
                False,  # shuffle color
                False, # config["hide_action"],
                hanalearn.AuxType.Null, # trinary
                replay_buffer,# if write_replay[player_idx] else None,
                1,  # mutli step, does not matter
                80,  # max_seq_len, default to 80
                0.999,  # gamma, does not matter
            )

            if config.get("llm_prior", None) is not None:
                llm_pkl = os.path.join(os.path.dirname(weight_file), "llm.pkl")
                llm_prior = pickle.load(open(llm_pkl, "rb"))
                # llm_prior = utils.load_and_process_llm_prior(
                #     config["llm_prior"], games[0], verbose=False
                # )
                # actor.set_llm_prior(llm_prior, [0.01875], config["pikl_beta"])
                actor.set_llm_prior(llm_prior, [config["pikl_lambda"]], config["pikl_beta"])

            thread_actor.append(actor)
        for i in range(len(thread_actor)):
            partners = thread_actor[:]
            partners[i] = None
            thread_actor[i].set_partners(partners)
        actors.append([thread_actor])

    context, threads = create_threads(num_thread, game_per_thread, actors, games)

    for runner in runners:
        runner.start()
    context.start()
    while replay_buffer.size() < num_game:
        time.sleep(0.2)

    print("dataset size:", replay_buffer.size())
    scores = []
    for g in games:
        s = g.last_episode_score()
        if s >= 0:
            scores.append(s)

    print(scores)
    print(
        "done about to return, avg score (%d game): %.2f"
        % (len(scores), np.mean(scores))
    )
    return replay_buffer, agent, context, games


def analyze(dataset, num_action, num_player=2, vdn=True):
    p0_p1 = np.zeros((num_action, num_action))

    for i in range(dataset.size() if vdn else dataset.size() // 2):
        epsd = dataset.get(i)
        action = epsd.action["a"]
        if num_player == 2 and action[0][0] == num_action:
            action = action[:, [1, 0]]

        for t in range(int(epsd.seq_len.item()) - 1):
            p0 = t % num_player
            p1 = (t + 1) % num_player
            a0 = action[t][p0]  # This indexing allows to avoid no-ops with vdn
            a1 = action[t + 1][p1]
            p0_p1[a0][a1] += 1

    # total_num_action = p0_p1.sum()
    # action_count = p0_p1.sum(1)
    # for i, count in enumerate(action_count):
    #     idx2action = get_idx2action(5, 5, 5)
        # print(f"{idx2action[i]}: {100 * count/total_num_action:.2f}")

    denom = p0_p1.sum()
    normed_p0_p1_global = p0_p1 / denom
    denom = p0_p1.sum(1, keepdims=True)
    normed_p0_p1 = p0_p1 / denom.clip(min=1e-3)

    return normed_p0_p1, normed_p0_p1_global, p0_p1


def get_idx2action(hand_size, num_color, num_rank):
    idx2action_ = []
    for i in range(hand_size):
        idx2action_.append(f"D{i+1}")
    for i in range(hand_size):
        idx2action_.append(f"P{i+1}")
    for i in range(num_color):
        idx2action_.append(f"C{i+1}")
    for i in range(num_rank):
        idx2action_.append(f"R{i+1}")
    return idx2action_


def render_plots(plot_funcs, rows, cols):
    fig = plt.figure(figsize=(cols * 8, rows * 8))
    ax = fig.subplots(rows, cols, squeeze=False)
    for i, func in enumerate(plot_funcs):
        func(fig, ax[i])
    print("render plots return")
    return fig, ax


def plot(mat, mat_gn, title, *, fig=None, ax=None, savefig=None, hle_game):
    print("rendering", title)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    cax = ax[0].matshow(mat)
    cax = ax[1].matshow(mat_gn)
    ax[0].set_title(title+"_global0")
    ax[1].set_title(title+"_global1")

    for axx in ax:
        ticks = get_idx2action(hle_game.hand_size(), hle_game.num_colors(), hle_game.num_ranks())
        axx.set_xticks(range(hle_game.max_moves()))
        axx.set_xticklabels(ticks)
        axx.set_yticks(range(hle_game.max_moves()))
        axx.set_yticklabels(ticks)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
    print("done")


def plot_factory(dataset, name, epoch, num_action, hle_game):
    normed_p0_p1, normed_p0_p1_global, _ = analyze(
        dataset, num_action, num_player=args.num_player
    )
    title = "%s_%d" % (name, epoch)

    def f(fig, ax):
        plot(normed_p0_p1, normed_p0_p1_global, title, fig=fig, ax=ax, hle_game=hle_game)

    return f


def render_root(sweep_root, include, savefig=None):
    models = common_utils.get_all_files(sweep_root, "pthw", contain="model0.pthw")
    models = sorted(models)
    if include is not None:
        print(include)
        models = [m for m in models if include in m]
    print(models)

    logs = parse_from_root(sweep_root, -1, -1, [], [], True)
    model_infos = {}
    for i, m in enumerate(models):
        name = m.split("/")[-2]
        print("M%2d: %s" % (i, name))
        model_infos["M%2d" % i] = {"path": m, "log": logs[name]}

    # get dataset
    datasets = {}
    contexts = []
    for k, v in model_infos.items():
        dset, _, context, games = create_dataset_new(
            v["path"], num_game=args.num_game, num_thread=args.num_thread, device=args.device
        )
        datasets[k] = dset
        # keep the contexts to hack off deadlock, ouch
        contexts.append(context)

    num_action = games[0].get_hle_game().max_moves()
    # render plots
    render_funcs = []
    for m_idx in model_infos.keys():
        #     print('generate rendering function for %s' % m_idx)
        epoch = model_infos[m_idx]["log"]["epoch"]
        dset = datasets[m_idx]
        name = shorten_name(model_infos[m_idx]["log"]["id"].split("/")[-2])
        render_function = plot_factory(
            dset, name, epoch, num_action, games[0].get_hle_game()
        )
        render_funcs.append(render_function)

    # print(len(render_funcs))
    cols = 2
    rows = len(render_funcs)
    render_plots(render_funcs, rows, cols)
    if savefig is None:
        plt.show()
    else:
        plt.tight_layout()
        print(f"saving image to {savefig}")
        plt.savefig(savefig)
        # not so elegant, but who cares
        os._exit(0)
    return


def render_run_folder(folder, max_num=-1, savefig=None):
    models = common_utils.get_all_files(folder, "pthw", contain="epoch")
    models = sorted(models)
    print(models)
    model_infos = {}
    for _, m in enumerate(models):
        name = m.split("/")[-1].split(".")[0].split("epoch")[1]
        model_infos[int(name)] = m

    # get dataset
    datasets = {}
    contexts = []
    # for k, v in model_infos.items():
    keys= sorted(list(model_infos.keys()))
    if max_num > 0:
        keys = keys[:max_num]

    for k in keys:
        v = model_infos[k]
        dset, _, context, games = create_dataset_new(v, device=args.device)
        datasets[k] = dset
        # keep the contexts to hack off deadlock, ouch
        contexts.append(context)

    num_action = games[0].get_hle_game().max_moves()
    # render plots
    render_funcs = []
    for model_epoch in keys:
        epoch = int(model_epoch)
        dset = datasets[model_epoch]
        render_function = plot_factory(
            dset, str(model_epoch), epoch, num_action, games[0].get_hle_game()
        )
        render_funcs.append(render_function)

    render_plots(render_funcs, rows=len(keys), cols=2)
    if savefig is None:
        plt.show()
    else:
        plt.tight_layout()
        print(f"saving image to {savefig}")
        plt.savefig(savefig)
        os._exit(0)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--run_folder", type=str, default=None)
    parser.add_argument("--max_num", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--include", type=str, default=None)
    args = parser.parse_args()

    if args.model is not None:
        dataset, _, _, games = create_dataset_new(
            args.model,
            num_thread=args.num_thread,
            num_game=args.num_game,
            device=args.device
        )
        num_action = games[0].get_hle_game().max_moves()
        print(num_action)
        normed_p0_p1, normed_p0_p1_global, _ = analyze(
            dataset, num_action, num_player=args.num_player
        )
        if args.output is None:
            args.output = args.model + f".action_matrix.png"

        plot(
            normed_p0_p1,
            normed_p0_p1_global,
            "action_matrix",
            savefig=args.output,
            hle_game=games[0].get_hle_game()
        )
        print(f"saving fig to {args.output}")
    elif args.root is not None:
        if args.output is None:
            args.output = os.path.join(args.root, "action_matrices.png")
        render_root(args.root, args.include, savefig=args.output)
    elif args.run_folder is not None:
        if args.output is None:
            args.output = os.path.join(args.run_folder, "action_matrices.png")
        render_run_folder(args.run_folder, max_num=args.max_num, savefig=args.output)
