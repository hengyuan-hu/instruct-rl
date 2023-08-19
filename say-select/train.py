import random

import numpy as np
import argparse
import tqdm

from ball_env import *
from q_agent import *


def train(
    vdn,
    ax,
    total_step,
    replay_size,
    lambda_,
    lr,
    batchsize,
    model,
    tokenizer,
    cache,
    seed=1,
    eps=0.1,
    epoch_len=20,
    use_fake_pred=False,
    log_q_history=[False, False],
    use_adam=[False, False],
    adam_eps=1e-5,
    prefix="",
):
    random.seed(seed)
    np.random.seed(seed + 1)

    print(f"{use_adam=}, {seed=}, {eps=}, {lambda_=}, {lr=}")
    if not isinstance(lambda_, Sequence):
        lambda_ = [lambda_ for _ in range(2)]

    env = BallEnv(5, 3, 2)
    prompt = "I should select the same number as my partner."
    agents = [
        LangQAgent(
            player_idx=0,
            num_action=env._num_ball,
            replay_size=replay_size,
            prompt=prompt,
            lambda_=lambda_[0],
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            use_fake_pred=use_fake_pred,
            log_q_history=log_q_history[0],
            use_adam=use_adam[0],
            adam_eps=adam_eps,
        ),
        LangQAgent(
            player_idx=1,
            num_action=env._num_ball + 1,
            replay_size=replay_size,
            prompt=prompt,
            lambda_=lambda_[1],
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            use_fake_pred=use_fake_pred,
            log_q_history=log_q_history[1],
            use_adam=use_adam[1],
            adam_eps=adam_eps,
        ),
    ]

    if vdn:
        agents[0].set_partner(agents[1])
        agents[1].set_partner(agents[0])

    max_score = -100
    best_agents = []

    train_scores = []
    eval_scores = []
    num_epoch = total_step // epoch_len
    for i in tqdm.tqdm(range(num_epoch)):
        agents, train_score = run(
            env,
            agents,
            epoch_len,
            lr=lr,
            gamma=1,
            eps=eps,
            is_train=True,
            print_intv=-1,
            batchsize=batchsize,
        )
        eval_score = eval_agents(env, agents)
        train_scores.append(train_score)
        eval_scores.append(eval_score)
        if eval_score >= max_score:
            max_score = eval_score
            best_agents = [agent.clone() for agent in agents]

    # title = None
    if ax is not None:
        ax.plot(train_scores, label=f"{prefix}train")
        ax.plot(eval_scores, linestyle="dashed", label=f"{prefix}eval")
        ax.title.set_text(
            f"{vdn=}, step={total_step},\nl={lambda_}, {lr=}, {batchsize=},\nseed={seed}, adam={use_adam}"
        )
        ax.legend()

    return best_agents, agents, None


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--total_step", type=int, default=50000)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lmd", type=float, default=0.25)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_run", type=int, default=3)

    args = parser.parse_args()
    return args


def get_run_name(args, save_dir):
    d = vars(args)
    name = []
    for k, v in d.items():
        name.append(f"{k}{v}")
    run_name = "_".join(name)
    run_id = 1
    while os.path.exists(os.path.join(save_dir, run_name)):
        run_name += f"_run{run_id}"
    return os.path.join(save_dir, run_name)


if __name__ == "__main__":
    import os
    import sys
    from llm import load_gptjlm_model
    import utils

    args = parse_args()

    tokenizer, model = load_gptjlm_model()
    cache = {}
    save_dir = get_run_name(args, "exps")
    logger_path = os.path.join(save_dir, "run.log")
    sys.stdout = utils.Logger(logger_path)
    print(f"saving to {save_dir}")

    fig, ax = utils.generate_grid(cols=args.num_run, rows=1, figsize=6)
    final_results = []
    human_policy_checks = []
    for j, seed in enumerate(list(range(args.seed, args.seed + args.num_run))):
        ax[j].set_ylim(ymin=-1.5, ymax=1.0)  # type: ignore
        best_agents, agents, title = train(
            True,
            ax[j],  # type: ignore
            args.total_step,
            replay_size=1000,
            lambda_=[0.0, args.lmd],
            lr=args.lr,
            epoch_len=100,
            batchsize=args.batchsize,
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            seed=seed,
            use_adam=[True, True],
            eps=args.eps,
        )
        is_human_policy = show_agent_conventions(best_agents[1])
        human_policy_checks.append(is_human_policy)
    final_results.append(f"human policy rate: {np.mean(human_policy_checks)}")

    fig.tight_layout()
    fig.savefig(f"{save_dir}/plot.png")
    for ret in final_results:
        print(ret)
