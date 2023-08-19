import os
import sys
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common_utils
import utils
from tools import parse_log


import matplotlib.pyplot as plt
plt.switch_backend("agg")


def render_folder(
    path,
    y_key,
    *,
    x_key=None,
    smooth_k=1,
    min_epoch=-1,
    max_epoch=-1,
    include=[],
    exclude=[],
    new_log=1,
    legend_loc=None,
    bbox_to_anchor=None,
    x_min=0,
    x_max=None,
    y_min=None,
    y_max=None,
    rename=lambda x: x,
    figsize=(8, 8),
    fig=None,
    ax0=None,
    savefig=None,
):
    logs = parse_log.parse_from_root(
        path, max_epoch, min_epoch, include, exclude, new_log
    )
    avg_logs = {}
    for k, v in logs.items():
        avg_logs[k] = common_utils.moving_average(v[y_key], smooth_k)
    avg_logs = parse_log.average_across_seed(avg_logs)

    if x_key is not None:
        avg_xs = {}
        for k, v in logs.items():
            avg_xs[k] = v[x_key]
        avg_xs = parse_log.average_across_seed(avg_xs)

    if fig is None:
        fig, ax0 = plt.subplots(1, 1, figsize=figsize)
        show = True
    else:
        show = False
    # for k, v in avg_logs.items():
    for k in sorted(avg_logs.keys()):
        v = avg_logs[k]
        mean, sem = v
        if x_key is None:
            x = list(range(len(mean)))
        else:
            x = avg_xs[k][0][: len(mean)]
        ax0.plot(x, mean, label=rename(k))

        lb = [m - sem for m, sem in zip(mean, sem)]
        ub = [m + sem for m, sem in zip(mean, sem)]
        ax0.fill_between(x, lb, ub, alpha=0.25)

    if legend_loc is None:
        ax0.legend(prop={"size": 15}, bbox_to_anchor=bbox_to_anchor)
    else:
        ax0.legend(loc=legend_loc, prop={"size": 15}, bbox_to_anchor=bbox_to_anchor)

    ax0.set_xlim(left=x_min, right=x_max)
    ax0.set_ylim(y_min, y_max)
    plt.tight_layout()
    if show:
        plt.show()
    if savefig:
        print(f"saving fig to {savefig}")
        plt.savefig(savefig)


def render_cross_play_matrix(logfile, figsize=(15, 15)):
    log = open(logfile, "r").readlines()
    cross_play = defaultdict(dict)
    for l in log:
        if len(l.split()) == 3 and l[0] == "M":
            m1, m2, score = l.split()
            score = float(score)
            cross_play[m1][m2] = score
            cross_play[m2][m1] = score

    num_models = len(cross_play)
    models = sorted(cross_play.keys(), key=lambda x: int(x[1:]))
    model_paths = utils.parse_first_dict(logfile)

    for m in models:
        print(m, ":", model_paths[m].split("/")[-2])

    table = np.zeros((num_models, num_models))

    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            table[i][j] = cross_play[mi][mj]

    def plot(mat):
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(mat)
        ax.set_xticks(range(num_models))
        ax.set_yticks(range(num_models))
        fig.colorbar(cax)
        return

    plot(table)


def generate_grid(cols, rows, figsize=7):
    fig = plt.figure(figsize=(cols * figsize, rows * figsize))
    ax = fig.subplots(rows, cols)
    return fig, ax
