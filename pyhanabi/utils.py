import os
import time
from collections import OrderedDict
import json
import pickle
import torch
import numpy as np

import r2d2
import ppo
from create import create_envs
import common_utils


def process_compiled_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        key = key.replace("_orig_mod.", "")
        new_state_dict[key] = val
    return new_state_dict


def get_seed(key):
    seeds = {
        'a': 48950461,
        'b': 27413430,
        'c': 88451596,
        'd': 29484194,
        'e': 38114476,
        'f': 76285101,
        'g': 53332048,
        'h': 63127331,
        'i': 8685811,
        'j': 44628626,
        'k': 1025640,
        'l': 14331352,
        'm': 10679909,
        'n': 16776711,
    }
    return seeds[key]


def print_gpu_info():
    omp = os.getenv("OMP_NUM_THREADS", None)
    print("OMP_NUM_THREADS", omp)

    gpu_id = os.getenv("SLURM_STEP_GPUS", None)
    if gpu_id is not None:
        print("gpu_id:", gpu_id)
    for i in range(torch.cuda.device_count()):
        print("device name:", torch.cuda.get_device_name(i))

    slurm_jobid = os.getenv("SLURM_JOBID", None)
    if slurm_jobid is not None:
        print("slurm job:", slurm_jobid)


def load_and_process_llm_prior(filename, env, verbose=True) -> dict[str, list[float]]:
    data = pickle.load(open(filename, "rb"))
    hle_game = env.get_hle_game()
    prior: dict[str, list[float]] = {}

    for hist in data:
        logits = []
        for action_idx in range(hle_game.max_moves()):
            move = hle_game.get_move(action_idx)
            move_key = move.to_string()
            if isinstance(data[hist][move_key], tuple):
                # print(data[hist][move_key])
                *_, logit = data[hist][move_key]
            else:
                logit = data[hist][move_key]
            logits.append(logit)
        logits.append(0)  # add logit for no-op
        prior[hist] = logits
        if verbose and "reveal" in hist and "color" in hist:
            print(f"hist: {hist}")
            for action_idx in range(hle_game.max_moves()):
                move = hle_game.get_move(action_idx)
                move_key = move.to_string()
                _, action, _ = data[hist][move_key]
                print(move.to_string(), action, f"{logits[action_idx]:.3f}")
            print("-" * 100)
    return prior


def parse_first_dict(lines):
    config_lines = []
    open_count = 0
    for i, l in enumerate(lines):
        if l.strip()[0] == "{":
            open_count += 1
        if open_count:
            config_lines.append(l)
        if l.strip()[-1] == "}":
            open_count -= 1
        if open_count == 0 and len(config_lines) != 0:
            break

    config = "".join(config_lines).replace("'", '"')
    config = config.replace("True", "true")
    config = config.replace("False", "false")
    config = config.replace("None", "null")
    config_dict = json.loads(config)
    return config_dict, lines[i + 1 :]  # type: ignore


def get_train_config(weight_file):
    if os.path.exists(f"{weight_file}.cfg"):
        cfg = pickle.load(open(f"{weight_file}.cfg", "rb"))
        return cfg

    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        assert False
        # return None

    lines = open(log, "r").readlines()
    cfg, _ = parse_first_dict(lines)
    return cfg


def flatten_dict(d, new_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_dict)
        else:
            new_dict[k] = v


def load_agent(weight_file, overwrite):
    """
    overwrite has to contain "device"
    """
    print("loading file from: ", weight_file)
    cfg = get_train_config(weight_file)

    hand_size = cfg.get("hand_size", 5)
    game = create_envs(
        1,
        1,
        cfg["num_player"],
        cfg.get("train_bomb", cfg.get("bomb", 0)),
        cfg["max_len"],
        hand_size=hand_size,
        num_color=cfg.get("num_color", 5),
        num_rank=cfg.get("num_rank", 5),
        num_hint=cfg.get("num_hint", 8),
    )[0]

    if "vdn" in cfg:
        # print("pikl_lambda:", cfg["pikl_lambda"])
        dqn_cfg = {
            "vdn": cfg.get("vdn", False),
            "multi_step": cfg["multi_step"],
            "gamma": cfg["gamma"],
            "device": overwrite["device"],
            "net": cfg.get("net", "publ-lstm"),
            "in_dim": game.feature_size(False),
            "hid_dim": cfg["rnn_hid_dim"],
            "out_dim": game.num_action(),
            "num_lstm_layer": cfg["num_lstm_layer"],
            "off_belief": cfg["off_belief"],
        }
        agent = r2d2.R2D2Agent(**dqn_cfg).to(overwrite["device"])
        load_weight(agent.online_net, weight_file, overwrite["device"])
        agent.sync_target_with_online()
    elif cfg.get("method", "") == "iql":
        # for legacy model
        dqn_cfg = {
            "vdn": False,
            "multi_step": cfg["multi_step"],
            "gamma": cfg["gamma"],
            "device": overwrite["device"],
            "net": cfg.get("net", "publ-lstm"),
            "in_dim": game.feature_size(False),
            "hid_dim": cfg["rnn_hid_dim"],
            "out_dim": game.num_action(),
            "num_lstm_layer": cfg["num_lstm_layer"],
            "off_belief": cfg["off_belief"],
        }
        agent = r2d2.R2D2Agent(**dqn_cfg).to(overwrite["device"])
        load_weight(agent.online_net, weight_file, overwrite["device"])
        agent.sync_target_with_online()
    else:
        ppo_cfg = {
            "ppo_clip": cfg["ppo_clip"],
            "ent_weight": cfg["ent_weight"],
            "multi_step": cfg["multi_step"],
            "gamma": cfg["gamma"],
            "device": overwrite["device"],
            "net_type": cfg["net"],
            "in_dim": game.feature_size(False),
            "hid_dim": cfg["rnn_hid_dim"],
            "out_dim": game.num_action(),
            "num_lstm_layer": cfg["num_lstm_layer"],
            "off_belief": cfg["off_belief"],
        }
        agent = ppo.PPOAgent(**ppo_cfg).to(overwrite["device"])
        load_weight(agent.policy_net, weight_file, overwrite["device"])
    return agent, cfg


def log_explore_ratio(games, expected_eps):
    explore = []
    for g in games:
        explore.append(g.get_explore_count())
    explore = np.stack(explore)
    explore = explore.sum(0)

    step_counts = []
    for g in games:
        step_counts.append(g.get_step_count())
    step_counts = np.stack(step_counts)
    step_counts = step_counts.sum(0)

    factor = []
    for i in range(len(explore)):
        if step_counts[i] == 0:
            factor.append(1.0)
        else:
            f = expected_eps / max(1e-5, (explore[i] / step_counts[i]))
            f = max(0.5, min(f, 2))
            factor.append(f)

    explore = explore.reshape((8, 10)).sum(1)
    step_counts = step_counts.reshape((8, 10)).sum(1)

    print("exploration:")
    for i in range(len(explore)):
        ratio = 100 * explore[i] / max(step_counts[i], 0.1)
        factor_ = factor[i * 10 : (i + 1) * 10]
        print(
            "\tbucket [%2d, %2d]: %5d, %5d, %2.2f%%: mean factor: %.2f"
            % (
                i * 10,
                (i + 1) * 10,
                explore[i],
                step_counts[i],
                ratio,
                np.mean(factor_),
            )
        )

    for g in games:
        g.reset_count()

    return factor


class Tachometer:
    def __init__(self):
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0

    def start(self):
        self.t = time.time()

    def lap(
        self, replay_buffer, num_train, factor, num_batch, target_ratio, current_sleep_time
    ) -> float:
        assert self.t is not None
        t = time.time() - self.t
        self.total_time += t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t

        sleep_time = 0
        if target_ratio is not None:
            actual_t = t - current_sleep_time * num_batch
            actual_ratio = (factor * num_train / actual_t) / buffer_rate
            print(f"ratio {train_rate/buffer_rate:.2f}, actual_ratio {actual_ratio:.2f}")
            target_train_rate = buffer_rate * target_ratio
            time_per_batch = actual_t / num_batch
            target_time_per_batch = train_rate / target_train_rate * time_per_batch
            sleep_time = target_time_per_batch - time_per_batch
            sleep_time = max(0, sleep_time)

        print(
            "Speed: train: %.1f, buffer_add: %.1f, buffer_size: %d"
            % (train_rate, buffer_rate, replay_buffer.size())
        )
        self.num_buffer = num_buffer
        self.num_train += num_train
        print("Total Time: %s, %ds" % (common_utils.sec2str(self.total_time), self.total_time))
        print(
            "Total Sample: train: %s, buffer: %s"
            % (
                common_utils.num2str(self.num_train),
                common_utils.num2str(self.num_buffer),
            )
        )
        return sleep_time


def load_weight(model, weight_file, device, *, state_dict=None):
    if state_dict is None:
        state_dict = torch.load(weight_file, map_location=device)
    source_state_dict = OrderedDict()
    target_state_dict = model.state_dict()

    for k, v in target_state_dict.items():
        if k not in state_dict:
            print("warning: %s not loaded [not found in file]" % k)
            state_dict[k] = v
        elif state_dict[k].size() != v.size():
            print(
                "warnning: %s not loaded\n[size mismatch %s (in net) vs %s (in file)]"
                % (k, v.size(), state_dict[k].size())
            )
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            print("removing: %s not used" % k)
        else:
            source_state_dict[k] = state_dict[k]

    model.load_state_dict(source_state_dict)
    return


# returns the number of steps in all actors
def get_num_acts(actors):
    total_acts = 0
    for actor in actors:
        if isinstance(actor, list):
            total_acts += get_num_acts(actor)
        else:
            total_acts += actor.num_act()
    return total_acts


def generate_explore_eps(base_eps, alpha, num_env):
    if num_env == 1:
        if base_eps < 1e-6:
            base_eps = 0
        return [base_eps]

    eps_list = []
    for i in range(num_env):
        eps = base_eps ** (1 + i / (num_env - 1) * alpha)
        if eps < 1e-6:
            eps = 0
        eps_list.append(eps)
    return eps_list


def generate_log_uniform(min_val, max_val, n):
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    uni = np.linspace(log_min, log_max, n)
    uni_exp = np.exp(uni)
    return uni_exp.tolist()


def get_agent_online_network(agent, is_ppo) -> torch.nn.Module:
    if is_ppo:
        return agent.policy_net
    return agent.online_net


import torch.distributed as dist


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()
