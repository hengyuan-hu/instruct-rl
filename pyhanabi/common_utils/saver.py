# model saver that saves top-k performing model
import os
import torch
import pickle

class TopkSaver:
    def __init__(self, save_dir, topk):
        self.save_dir = save_dir
        self.topk = topk
        self.worse_perf = -float("inf")
        self.worse_perf_idx = 0
        self.perfs = [self.worse_perf]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save(self, state_dict, perf, *, save_latest=False, force_save_name=None, config=None):
        if force_save_name is not None:
            weight_name = os.path.join(self.save_dir, "%s.pthw" % force_save_name)
            torch.save(state_dict, weight_name)
            if config is not None:
                pickle.dump(config, open(f"{weight_name}.cfg", "wb"))

        if save_latest:
            weight_name = os.path.join(self.save_dir, "latest.pthw")
            torch.save(state_dict, weight_name)
            if config is not None:
                pickle.dump(config, open(f"{weight_name}.cfg", "wb"))

        if perf is None:
            return False

        if perf <= self.worse_perf:
            return False

        weight_name = os.path.join(self.save_dir, "model%i.pthw" % self.worse_perf_idx)
        torch.save(state_dict,  weight_name)
        if config is not None:
            pickle.dump(config, open(f"{weight_name}.cfg", "wb"))

        if len(self.perfs) < self.topk:
            self.perfs.append(perf)
            return True

        # neesd to replace
        self.perfs[self.worse_perf_idx] = perf
        worse_perf = self.perfs[0]
        worse_perf_idx = 0
        for i, perf in enumerate(self.perfs):
            if perf < worse_perf:
                worse_perf = perf
                worse_perf_idx = i

        self.worse_perf = worse_perf
        self.worse_perf_idx = worse_perf_idx
        return True
