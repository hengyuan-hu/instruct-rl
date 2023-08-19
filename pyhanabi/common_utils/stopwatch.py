from contextlib import contextmanager
from collections import defaultdict
# from datetime import datetime
import time
import numpy as np
import tabulate


def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


class Stopwatch:
    """stop watch in MS"""
    def __init__(self):
        self.times = defaultdict(list)

    def reset(self):
        self.times = defaultdict(list)

    @contextmanager
    def time(self, key):
        t = time.time()
        yield

        self.times[key].append(1000 * (time.time() - t))  # record in ms

    def summary(self):
        headers = ["name", "num", "t/call (ms)", "%"]
        total = 0
        times = {}
        for k, v in self.times.items():
            sum_t = np.sum(v)
            mean_t = sum_t / len(v)
            times[k] = (len(v), sum_t, mean_t)
            total += sum_t

        print("Timer Info:")
        rows = []
        for k, (num, sum_t, mean_t) in times.items():
            rows.append([k, f"{num:.1f}", f"{mean_t:.1f}", f"{100 * sum_t / total:.1f}"])
        print(tabulate.tabulate(rows, headers=headers))
        self.reset()
