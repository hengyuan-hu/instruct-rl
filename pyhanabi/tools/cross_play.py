import argparse
import os
import sys
import pprint
import itertools
from collections import defaultdict
import numpy as np
import torch


lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
import common_utils


def filter_include(entries, includes):
    if not isinstance(includes, list):
        includes = [includes]
    keep = []
    for entry in entries:
        for include in includes:
            if include not in entry:
                break
        else:
            keep.append(entry)
    return keep


def filter_exclude(entries, excludes):
    if not isinstance(excludes, list):
        excludes = [excludes]
    keep = []
    for entry in entries:
        for exclude in excludes:
            if exclude in entry:
                break
        else:
            keep.append(entry)
    return keep


def cross_play(models, num_player, num_game, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combs = list(itertools.combinations_with_replacement(models, num_player))
    perfs = defaultdict(list)
    for comb in combs:
        num_model = len(set(comb))
        score = evaluate_saved_model(comb, num_game, seed, 0, device=device)[0]
        perfs[num_model].append(score)

    for num_model, scores in perfs.items():
        print(
            f"#model: {num_model}, #groups {len(scores)}, "
            f"score: {np.mean(scores):.2f} "
            f"+/- {np.std(scores) / np.sqrt(len(scores) - 1):.2f}"
        )


parser = argparse.ArgumentParser()
parser.add_argument("--root", default=None, type=str, required=True)
parser.add_argument("--num_player", default=2, type=int)
parser.add_argument("--include", default=None, type=str, nargs="+")
parser.add_argument("--exclude", default=None, type=str, nargs="+")

args = parser.parse_args()

models = common_utils.get_all_files(args.root, "model0.pthw")
if args.include is not None:
    models = filter_include(models, args.include)
if args.exclude is not None:
    models = filter_exclude(models, args.exclude)

pprint.pprint(models)
cross_play(models, args.num_player, 1000, 1)
