# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import numpy as np
import torch

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model


# main program
parser = argparse.ArgumentParser()
parser.add_argument("--weight1", default=None, type=str, required=True)
parser.add_argument("--weight2", default=None, type=str)
parser.add_argument("--weight3", default=None, type=str)
parser.add_argument("--num_player", default=2, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--bomb", default=0, type=int)
parser.add_argument("--num_game", default=5000, type=int)
parser.add_argument(
    "--num_run",
    default=1,
    type=int,
    help="num of {num_game} you want to run, i.e. num_run=2 means 2*num_game",
)
parser.add_argument("--overwrite1", default=None, type=str)
parser.add_argument("--overwrite2", default=None, type=str)

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.num_player == 2:
    if args.weight2 is None:
        args.weight2 = args.weight1
    weight_files = [args.weight1, args.weight2]
elif args.num_player == 3:
    if args.weight2 is None:
        weight_files = [args.weight1 for _ in range(args.num_player)]
    else:
        weight_files = [args.weight1, args.weight2, args.weight3]
else:
    assert False

overwrites = []
for ovwt in [args.overwrite1, args.overwrite2]:
    if ovwt is None:
        overwrites.append({})
    else:
        overwrite = {}
        for kv in args.overwrite.split(","):
            k, v = kv.split("=")
            try:
                v = float(v)
            except:
                pass
            overwrite[k] = v
        overwrites.append(overwrite)


_, _, _, scores, actors, _ = evaluate_saved_model(
    weight_files,
    args.num_game,
    args.seed,
    args.bomb,
    num_run=args.num_run,
    device=device,
    overwrites=overwrites
)
non_zero_scores = [s for s in scores if s > 0]
print(f"non zero mean: {np.mean(non_zero_scores):.3f}")
print(f"bomb out rate: {100 * (1 - len(non_zero_scores) / len(scores)):.2f}%")
# print(scores)

# 4 numbers represent: [none, color, rank, both] respectively
card_stats = []
for g in actors:
    card_stats.append(g.get_played_card_info())
card_stats = np.array(card_stats).sum(0)

print("knowledge of cards played:")
total = sum(card_stats)
for i, ck in enumerate(["none", "color", "rank", "both"]):
    print(f"{ck}: {card_stats[i]}, {int(100 * card_stats[i] / total)}")
