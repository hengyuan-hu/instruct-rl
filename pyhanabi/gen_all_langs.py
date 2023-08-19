import os
import itertools
import argparse
import pickle

import set_path
import torch
import rela
import hanalearn


# RYGWB from hle
Colors = ["red", "yellow", "green", "white", "blue"]
Ranks = ["1", "2", "3", "4", "5"]


def gen_lang_for_history_item(env, hand_positions):
    game = env.get_hle_game()
    history_item2lang: dict[str, str] = {}
    history_item2lang["[null]"] = "My partner did nothing"
    for uid in range(game.max_moves()):
        type2action_name = {
            hanalearn.MoveType.Play: "played",
            hanalearn.MoveType.Discard: "discarded",
        }
        type2hint_type = {
            hanalearn.MoveType.RevealColor: "color",
            hanalearn.MoveType.RevealRank: "rank",
        }

        move = game.get_move(uid)
        print(f"move: {move.to_string()}")
        history_item = hanalearn.HanabiHistoryItem(move)
        history_item.player = 1

        if move.move_type() in type2action_name:
            key = history_item.to_lang_key()
            action_name = type2action_name[move.move_type()]
            pos = hand_positions[move.card_index()]
            if "A" in hand_positions:
                lang = f"My partner {action_name} their card at position {pos}"
            else:
                lang = f"My partner {action_name} their '{pos}' card"
            history_item2lang[key] = lang

        elif move.move_type() in type2hint_type:
            hint_type = type2hint_type[move.move_type()]
            for num_hinted_pos in range(1, len(hand_positions) + 1):
                for hinted_poses in itertools.combinations(hand_positions, num_hinted_pos):
                    bitmask = 0
                    for hinted_pos in hinted_poses:
                        bitmask |= 1 << hand_positions.index(hinted_pos)
                    history_item.reveal_bitmask = bitmask

                    lang_pos = ""
                    for i, hinted_pos in enumerate(hinted_poses):
                        if i == 0:
                            lang_pos += f"{hinted_pos}"
                        elif i == len(hinted_poses) - 1:
                            lang_pos += f" and {hinted_pos}"
                        else:
                            lang_pos += f", {hinted_pos}"

                    card = "card" if len(hinted_poses) == 1 else "cards"
                    if "A" in hand_positions:
                        lang_pos = f"{card} at position {lang_pos}"
                    else:
                        lang_pos = f"{lang_pos} {card}"

                    if move.move_type() == hanalearn.MoveType.RevealColor:
                        hinted_val = Colors[move.color()]
                    else:
                        hinted_val = f"{Ranks[move.rank()]}"

                    key = history_item.to_lang_key()
                    lang = f"My partner told me that the {hint_type} of my {lang_pos} is {hinted_val}"
                    history_item2lang[key] = lang

    return history_item2lang


def gen_lang_for_all_moves(env, hand_positions, full_action_list):
    game = env.get_hle_game()
    move2lang: dict[str, str] = {}
    # play/discard
    assert len(hand_positions) == game.hand_size()
    for idx, pos in enumerate(hand_positions):
        name_types = [
            ("play", hanalearn.MoveType.Play),
            ("discard", hanalearn.MoveType.Discard),
        ]
        for action_name, action_type in name_types:
            move = hanalearn.HanabiMove(
                action_type,  # move_type
                idx,  # card_index
                1,  # target_offset, fixed to 1 because we only have 2 player
                -1,  # color
                -1,  # rank
            )
            print(hand_positions)
            if "A" in hand_positions:
                lang = f" {action_name} my card at position {pos}"
            else:
                lang = f" {action_name} my '{pos}' card"
            move2lang[move.to_string()] = lang

    # print(f"num_color: {game.num_colors()}")
    # print(f"num_rank: {game.num_ranks()}")
    # hints
    for i in range(game.num_colors()):
        move = hanalearn.HanabiMove(
            hanalearn.MoveType.RevealColor,  # move_type
            -1,  # card_index
            1,  # target_offset, fixed to 1 because we only have 2 player
            i,  # color
            -1,  # rank
        )
        if full_action_list:
            lang = f" hint 'color {Colors[i]}' to my partner"
        else:
            lang = f" hint color to my partner"
        move2lang[move.to_string()] = lang

    for i in range(game.num_ranks()):
        move = hanalearn.HanabiMove(
            hanalearn.MoveType.RevealRank,  # move_type
            -1,  # card_index
            1,  # target_offset, fixed to 1 because we only have 2 player
            -1,  # color
            i,  # rank
        )
        if full_action_list:
            lang = f" hint 'rank {Ranks[i]}' to my partner"
        else:
            lang = f" hint rank to my partner"
        move2lang[move.to_string()] = lang
    return move2lang


def create_game(hand_size, num_color, num_rank):
    params = {
        "players": str(2),
        "seed": str(1),
        "bomb": str(0),
        "hand_size": str(hand_size),
        "colors": str(num_color),
        "ranks": str(num_rank),
        "random_start_player": str(0),  # bool 0=false, start from 0
    }
    env = hanalearn.HanabiEnv(params, -1, False)
    return env


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_dir", type=str, default="exps/lang")
    parser.add_argument("--num_color", type=int, default=5)
    parser.add_argument("--num_rank", type=int, default=5)
    parser.add_argument("--hand_positions", type=str, default="A,B,C,D,E")
    parser.add_argument("--full_action_list", type=int, default=0)
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--rerun", type=int, default=0)

    args = parser.parse_args()
    args.hand_positions = args.hand_positions.split(",")
    args.hand_size = len(args.hand_positions)
    return args


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env = create_game(args.hand_size, args.num_color, args.num_rank)

    move2lang = gen_lang_for_all_moves(env, args.hand_positions, args.full_action_list)
    print("Current Actions")
    for k, v in move2lang.items():
        print(f"{k} --> {v}")
    print("=" * 100)

    hist2lang = gen_lang_for_history_item(env, args.hand_positions)
    print("History Items")
    for k, v in hist2lang.items():
        print(f"{k} --> {v}")
    print("=" * 100)

    hist_file = os.path.join(args.save_dir, f"{args.save_prefix}hist.pkl")
    print(f"saving hist to {hist_file}")
    move_file = os.path.join(args.save_dir, f"{args.save_prefix}move.pkl")
    print(f"saving move to {move_file}")
    pickle.dump(hist2lang, open(hist_file, "wb"))
    pickle.dump(move2lang, open(move_file, "wb"))
