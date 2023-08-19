import abc
import os
import sys
import pickle
import torch

pyhanabi = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyhanabi)

import set_path
import r2d2
from utils import load_agent
from game_state import HleGameState
from sparta import *


class Agent(abc.ABC):
    @abc.abstractmethod
    def init_and_get_h0(self, state: HleGameState):
        pass

    @abc.abstractmethod
    def observe_and_maybe_act(self, state: HleGameState, hid):
        pass


class PiklAgent(Agent):
    def __init__(self, model_path, override=None):
        self.model_path = model_path
        default_override = {"device": "cpu"}
        if override is not None:
            default_override.update(override)

        self.agent, self.cfg = load_agent(model_path, default_override)
        self.agent.train(False)

        self.pikl_lambda = self.cfg["pikl_lambda"]
        print(f"pikl_lambda: {self.pikl_lambda}")
        self.pikl_beta = self.cfg["pikl_beta"]
        llm_file = os.path.join(os.path.dirname(model_path), "llm.pkl")
        self.llm_prior = pickle.load(open(llm_file, "rb"))
        assert self.pikl_beta == 1

    def init_and_get_h0(self, state: HleGameState):
        h0 = self.agent.get_h0(1)
        return h0

    def observe_and_maybe_act(self, state: HleGameState, hid):
        priv_s, publ_s, legal_move, hist = state.observe()
        assert isinstance(self.agent, r2d2.R2D2Agent)

        if hist is None or (not state.is_my_turn()):
            hist_key = "[null]"
        else:
            hist_key = hist.to_lang_key()

        pikl_lambda = torch.tensor([self.pikl_lambda])
        llm_prior = torch.tensor(self.llm_prior[hist_key]).unsqueeze(0)

        action, new_hid, _ = self.agent.pikl_act(
            priv_s, publ_s, legal_move, hid, pikl_lambda, llm_prior
        )

        move = None
        if state.is_my_turn():
            move = state.hle_game.get_move(action.item())
        return move, new_hid
