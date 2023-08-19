from contextlib import contextmanager
import time
import os
import sys
import matplotlib.pyplot as plt


def generate_grid(cols, rows, figsize=7):
    fig = plt.figure(figsize=(cols * figsize, rows * figsize))
    ax = fig.subplots(rows, cols)
    return fig, ax


@contextmanager
def timer(msg):
    t = time.time()
    yield
    print(f"{msg} takes: {time.time() - t:.2f}s")


class Logger:
    def __init__(self, path, mode="w"):
        assert mode in {"w", "a"}, "unknown mode for logger %s" % mode
        self.terminal = sys.stdout
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if mode == "w" or not os.path.exists(path):
            self.log = open(path, "w")
        else:
            self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # for python 3 compatibility.
        pass
