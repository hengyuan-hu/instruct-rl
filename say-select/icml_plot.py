# This file is meant to be run as interactive cells in vscode python mode

# %%

import numpy as np
from ball_env import *
from q_agent import *
from train import train
from llm import *
from utils import timer, generate_grid


# %%
tokenizer, model = load_gptjlm_model()
cache = {}


# %%
fig, ax = generate_grid(cols=4, rows=1, figsize=4)


with timer("run0"):
    # TODO: dynamic learning rate helps
    best_agents0, agents0, _ = train(
        True,
        ax[0],
        50000,
        replay_size=1000,
        lambda_=[0.0, 0.0],
        lr=0.02,
        epoch_len=100,
        batchsize=64,
        eps=0.15,
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        seed=9,
        log_q_history=[False, False],
        use_adam=[True, True],
    )


def show_agent_conventions(agent):
    assert agent._player_idx == 1
    obs_actions = agent.get_policy()

    def sort_func(obs_action):
        obs, _ = obs_action
        past_action = obs[-1]
        if len(past_action) == 1:
            return past_action[0] - 100
        return past_action[1] * 10 + past_action[0]

    obs_actions = sorted(obs_actions, key=sort_func)
    last_actions = None
    is_human_policy = True

    policy_mat: list[list[str]] = [["" for _ in range(5)] for _ in range(6)]
    for obs, action in obs_actions:
        if last_actions is not None:
            if len(obs[2]) > len(last_actions):
                print("-" * 20)
            elif obs[2][-1] != last_actions[-1] and len(obs[2]) > 1:
                print("-" * 20)
        if action == 5:
            print(f"{[agent._action_names[a] for a in obs[2]]} -> quit")
            if len(obs[2]) != 2 or obs[2][0] != obs[2][1]:
                is_human_policy = False
        else:
            print(
                f"{[agent._action_names[a] for a in obs[2]]} -> {agent._action_names[action]}"
            )
            if obs[2][-1] != action:
                is_human_policy = False

        if action == "5":
            action_str = "0"
        else:
            action_str = agent._action_names[action]

        action_str = int(action_str)
        if len(obs[2]) == 1:
            policy_mat[0][obs[2][0]] = action_str
        else:
            policy_mat[obs[2][0] + 1][obs[2][1]] = action_str

        last_actions = obs[2]
    return is_human_policy, policy_mat


env = BallEnv(5, 3, 2)
score = eval_agents(env, best_agents0)
print(score)
is_human_policy, policy_mat = show_agent_conventions(best_agents0[1])
print(f"{is_human_policy=}")
print("@" * 50)

# %%
a_t2_labels = ["n/a", "1", "2", "3", "4", "5"]
a_t1_labels = ["1", "2", "3", "4", "5"]

instruct_policy = [
    [1, 2, 3, 4, 5],
    [0, 2, 3, 4, 5],
    [1, 0, 3, 4, 5],
    [1, 2, 0, 4, 5],
    [1, 2, 3, 0, 5],
    [1, 2, 3, 4, 0],
]

q_policy1 = [
    [3, 4, 2, 5, 1],
    [3, 4, 2, 0, 5],
    [3, 4, 5, 0, 1],
    [0, 4, 2, 0, 1],
    [3, 0, 2, 0, 1],
    [3, 4, 2, 0, 1],
]

q_policy2 = [
    [1, 4, 5, 2, 3],
    [4, 4, 0, 2, 3],
    [1, 4, 0, 5, 3],
    [1, 4, 0, 2, 5],
    [1, 5, 0, 2, 3],
    [1, 4, 0, 2, 3],
]

# avoid type3 font
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
plt.rc("ytick", labelsize=20)
plt.rc("axes", labelsize=20)
plt.rc("axes", titlesize=20)
from matplotlib import colors
import seaborn as sns

# print(sns.color_palette("Blues"))
print(len(sns.color_palette("rocket")))
# Unicorn
# color = ["#edda95", "#c5b8a5", "#b4c9c8", "#faead3", "#8696a7", "#dadad8", "#a27e7e"][1:]
# Melancholy
# color = ["#20233c", "#264b75", "#8696a7", "#c9c0d3", "#c1cbd7", "#eee5f8", "#ebf5f0"][1:]
color = ["#edda95", "#8696a7", "#c9c0d3", "#c1cbd7", "#eee5f8", "#ebf5f0"]
blues = sns.color_palette("Blues")[:5]
yellows = sns.color_palette("YlOrBr")[:5]
color1 = ["#edda95"] + blues
color2 = ["#8696a7"] + yellows
cmap = colors.ListedColormap(color1)


def plot(policy_mat, ax, title):
    x = np.array(policy_mat)
    print(x)
    ax.set_yticks(range(len(a_t2_labels)))
    ax.set_yticklabels(a_t2_labels)
    ax.set_xticks(range(len(a_t1_labels)))
    ax.set_xticklabels(a_t1_labels)
    ax.tick_params(axis="both", which="major", pad=0)

    ax.set_xticks(np.arange(-0.5, len(a_t1_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(a_t2_labels), 1), minor=True)

    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.grid(False)
    ax.matshow(np.array(policy_mat), cmap=cmap)

    x_ind_array = np.arange(-0.16, 4, 1.0)
    y_ind_array = np.arange(0.17, 6, 1.0)
    for yid, yy in enumerate(y_ind_array):
        for xid, xx in enumerate(x_ind_array):
            val = policy_mat[yid][xid]
            if val == 0:
                val = "Q"
            else:
                val = str(val)
            ax.text(xx, yy, val, size=20, color="black")

    # ax.text(x[0][0], y[0][0], "Q", size=20)
    ax.tick_params(which="minor", top=False, bottom=False, left=False)
    ax.tick_params(which="major", top=False, bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, y=-0.1)


fig, ax = generate_grid(rows=1, cols=3, figsize=5)
plot(q_policy1, ax[0], "Q policy (1)")
plot(q_policy2, ax[1], "Q policy (2)")
plot(instruct_policy, ax[2], "InstructQ policy")

# %%
