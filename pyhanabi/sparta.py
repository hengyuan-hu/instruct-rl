import os
import sys
import time
import pickle
import argparse
import pprint
from typing import Optional

import numpy as np
from create import create_envs
import torch
import rela
import hanalearn

import r2d2
from belief_model import ARBeliefModel
import utils
import common_utils


__GLOBAL_BATCH_RUNNER_CACHE = {}
__GLOBAL_MODEL_CACHE = {}


def get_batch_runner(model_file, model, device, methods):
    if model_file in __GLOBAL_BATCH_RUNNER_CACHE:
        return __GLOBAL_BATCH_RUNNER_CACHE[model_file]

    batch_runner = rela.BatchRunner(model, device)
    for method, bsz in methods.items():
        batch_runner.add_method(method, bsz)
    batch_runner.start()

    __GLOBAL_BATCH_RUNNER_CACHE[model_file] = batch_runner
    return batch_runner


def get_model(model_file, model_type, device):
    assert model_type in ["policy", "belief"]
    if model_file in __GLOBAL_MODEL_CACHE:
        return __GLOBAL_MODEL_CACHE[model_file]

    if model_type == "policy":
        model, _ = utils.load_agent(model_file, {"device": device})
    else:
        model = ARBeliefModel.load(model_file, device, 5)

    __GLOBAL_MODEL_CACHE[model_file] = (model)
    return model


class Sparta:
    def __init__(
        self,
        player_idx,
        device,
        bp_file,
        belief_file,
        seed,
        do_search=True,
        llm_lambda=0.0,
        qre_lambda=0.0,
        prior=None,
    ):
        self.player_idx = player_idx
        self.device = device
        self.do_search = do_search
        self.qre_lambda = qre_lambda

        self.bp_model = get_model(bp_file, "policy", device)
        self.bp_runner = get_batch_runner(bp_file, self.bp_model, device, {"act": 1000})

        # llm related
        self.llm_lambda = torch.tensor([llm_lambda]).to(device)
        self.llm_prior: dict[str, torch.Tensor] = {}
        if prior is not None:
            llm_prior = pickle.load(open(prior, "rb"))
            for k, v in llm_prior.items():
                self.llm_prior[k] = torch.tensor(v, dtype=torch.float32)

        self.belief_model: Optional[ARBeliefModel] = None
        if belief_file is not None:
            self.belief_model = get_model(belief_file, "belief", device)

        self.rng = np.random.default_rng(seed=seed)

        self.all_players = []
        self.bp_hid: dict[str, torch.Tensor] = {}
        self.pre_act_bp_hid: dict[str, torch.Tensor] = {}
        self.belief_hid: dict[str, torch.Tensor] = {}
        self.reset()

    def get_search_player(self):
        bp_hid = common_utils.to_device(self.pre_act_bp_hid, "cpu", detach=True)
        player = hanalearn.SearchPlayer(self.player_idx, self.bp_runner, bp_hid)
        if len(self.llm_prior) > 0:
            player.set_llm_prior(self.llm_prior, self.llm_lambda.cpu().item(), 1.0)
        return player

    def reset(self):
        self.bp_hid = common_utils.to_device(self.bp_model.get_h0(1), self.device)
        self.pre_act_bp_hid = self.bp_hid

        if self.belief_model is not None:
            self.belief_hid = common_utils.to_device(self.belief_model.get_h0(), self.device)

    def set_all_players(self, players):
        self.all_players = players

    def act(self, state, game, num_search):
        self.pre_act_bp_hid = self.bp_hid

        obs, card_count, _ = hanalearn.sparta_observe(state, self.player_idx)
        # the unsqueezed dim is used as seq_dim for rnn, but batch_dim for bp
        priv_s = obs["priv_s"].to(self.device).unsqueeze(0)

        move_scores = None
        if self.do_search and state.cur_player() == self.player_idx:
            assert self.belief_model is not None
            legal_moves = state.legal_moves(self.player_idx)
            search_per_move = num_search // len(legal_moves)
            print(num_search, len(legal_moves), search_per_move)

            num_sample = search_per_move * 2
            samples, self.belief_hid = self.belief_model.sample(priv_s, self.belief_hid, num_sample)
            samples = samples.cpu().squeeze(1)

            my_hand = state.hands()[self.player_idx]
            filtered_samples = hanalearn.filter_sample(samples, card_count, game, my_hand)
            print(f"filter sampled hands: {num_sample} -> {len(filtered_samples)}")

            if len(filtered_samples) > search_per_move:
                filtered_samples = filtered_samples[:search_per_move]
                print(common_utils.get_mem_usage(" before search"))
                move_scores = self.search(state, filtered_samples)
                print(common_utils.get_mem_usage(" after search"))
            elif len(filtered_samples) < 0.5 * search_per_move:
                print("too few samples, abort search")
        else:
            if self.belief_model is not None:
                self.belief_hid = self.belief_model.observe(priv_s, self.belief_hid)

        if isinstance(self.bp_model, r2d2.R2D2Agent):
            publ_s = priv_s[:, 125:]
            legal_move = obs["legal_move"].unsqueeze(0).to(self.device)

            if len(self.llm_prior) == 0 or state.cur_player() != self.player_idx:
                action, self.bp_hid, extra = self.bp_model.greedy_act(
                    priv_s, publ_s, legal_move, self.bp_hid
                )
                adv = extra["adv"].squeeze(0)
            else:
                hist_move = hanalearn.get_last_non_deal_move_from_state(state, self.player_idx)
                hist_key = hist_move.to_lang_key() if hist_move is not None else "[null]"
                prior = self.llm_prior[hist_key].to(self.device).unsqueeze(0)  # expand batch
                action, self.bp_hid, extra = self.bp_model.pikl_act(
                    priv_s,
                    publ_s,
                    legal_move,
                    self.bp_hid,
                    self.llm_lambda,
                    prior,
                )
                adv = extra["legal_adv"].squeeze(0)

            action = action.item()
        else:
            assert False

        if move_scores is None:
            return action

        action = self.select_and_log_action(game, action, move_scores, adv)
        if state.cur_player() == self.player_idx:
            print(f"player {self.player_idx}, move: {game.get_move(action).to_string()}")

        return action

    def search(self, state, samples):
        print("search per move:", len(samples))
        sim_seeds = self.rng.integers(low=1, high=int(1e8), size=len(samples))
        search_players = [player.get_search_player() for player in self.all_players]

        legal_moves = state.legal_moves(self.player_idx)
        scores = []
        # for move in legal_moves:
        #     score = hanalearn.search_move(
        #         state, move, samples, sim_seeds, self.player_idx, search_players
        #     )
        #     print(move.to_string(), score)
        #     scores.append(score)

        scores = hanalearn.parallel_search_moves(
            state, legal_moves, samples, sim_seeds, self.player_idx, search_players
        )
        move_scores = list(zip(legal_moves, scores))
        return move_scores

    def select_and_log_action(self, game, bp_action, move_scores, adv) -> int:
        best_action = -1
        best_combined = -1
        bp_combined = -1

        combined_scores = {}

        for move, score in move_scores:
            move_uid = game.get_move_uid(move)
            combined = score

            if adv is not None:
                combined += self.qre_lambda * adv[move_uid].item()

            combined_scores[move.to_string()] = combined

            if move_uid == bp_action:
                bp_combined = combined
                if self.qre_lambda == 0:
                    bp_combined += 0.05

            if combined > best_combined:
                best_combined = combined
                best_action = move_uid

        if bp_combined < best_combined:
            action = best_action
        else:
            action = bp_action

        for move, score in move_scores:
            move_uid = game.get_move_uid(move)
            info = f"{move.to_string()}: {score:.2f}, "
            if adv is not None:
                move_adv = adv[move_uid]
                info += f"adv: {move_adv:.2f}, "

            combined_score = combined_scores[move.to_string()]
            info += f"sum: {combined_score:.2f}"
            if move_uid == bp_action:
                info = f"{info}, (bp)"
            if move_uid == best_action:
                info = f"{info}, (best)"
            if move_uid == action:
                info = f"{info}, (selected)"
            print(info)

        return action


def run_game(seed, players):
    game = create_envs(
        num_env=1,
        seed=seed,
        num_player=len(players),
        bomb=0,
        max_len=-1,
    )[0]
    game.reset()
    for player in players:
        player.reset()

    step = 0
    with torch.no_grad():
        while not game.terminated():
            print(f"===== Step{step} =====")
            print(game.get_hle_state().to_string())

            t = time.time()
            actions = []
            for player in players:
                actions.append(player.act(game.get_hle_state(), game.get_hle_game(), 10000))
            action = actions[game.get_current_player()]
            move = game.get_move(action)
            print(f">>>>>move: {move.to_string()}")
            game.step(move)
            step += 1
            print(f"time {time.time() - t:.1f}")

    print(f"game end, len: {step}, score: {game.get_score()}")
    return game.get_score()


iql_rank = "../models/icml/iql_rank/iql1_pkr_load_pikl_lambda0.15_seeda_num_epoch50/model0.pthw"
rank_prior = "../models/icml/iql_rank/iql1_pkr_load_pikl_lambda0.15_seeda_num_epoch50/llm.pkl"
iql = "../models/icml/iql_baseline/iql1_pkc_load_pikl_lambda0.0_seeda_num_epoch50/model0.pthw"
iql_belief = "../models/icml/iql_baseline/belief_lr_lr2.5e-4/latest.pthw"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_dir", type=str, default="exps/sparta")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--imagined_lambda", type=float, default=None)
    parser.add_argument("--qre_lambda", type=float, default=0)

    args = parser.parse_args()
    pprint.pprint(vars(args))

    logger_path = os.path.join(args.save_dir, f"game{args.seed}.log")
    sys.stdout = common_utils.Logger(logger_path, print_to_stdout=True)
    rng = np.random.default_rng(seed=args.seed + 1)
    player_seeds = rng.integers(low=1, high=int(1e4), size=3)

    _, p0_cfg = utils.load_agent(iql_rank, {"device": "cpu"})
    true_pikl_lambda = p0_cfg["pikl_lambda"]
    print("true pikl lambda:", true_pikl_lambda)
    if args.imagined_lambda is None:
        print("set imagined lambda to true pikl lambda")
        args.imagined_lambda = true_pikl_lambda

    players = [
        Sparta(
            player_idx=0,
            device="cuda",
            bp_file=iql_rank,
            belief_file=None,
            seed=player_seeds[0],
            do_search=False,
            llm_lambda=true_pikl_lambda,
            prior=rank_prior,
        ),
        Sparta(
            player_idx=1,
            device="cuda",
            bp_file=iql,
            belief_file=iql_belief,
            seed=player_seeds[1],
            do_search=True,
            # imagine that partner may use this lambda, we also use the same value
            llm_lambda=args.imagined_lambda,
            prior=rank_prior,
            qre_lambda=args.qre_lambda,
        ),
    ]
    imagined_partner = Sparta(
        player_idx=0,
        device="cuda",
        bp_file=iql,
        belief_file=None,
        seed=player_seeds[2],
        do_search=False,
        llm_lambda=args.imagined_lambda,
        prior=rank_prior,
    )
    players[1].set_all_players([imagined_partner, players[1]])

    t = time.time()
    score = run_game(args.seed, players)
    print(f"total time: {time.time() - t:.2f}")
    print(f"seed {args.seed}: {score}")
