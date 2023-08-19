import os
import pickle
import numpy as np

from create import *
import rela
import utils
import r2d2


def evaluate(
    agents,
    num_game,
    seed,
    bomb,
    *,
    num_player=None,
    num_thread=10,
    max_len=80,
    device="cuda:0",
    hand_size=5,
    num_color=5,
    num_rank=5,
    num_hint=8,
    llm_priors=None,
    pikl_lambdas=None,
    pikl_betas=None,
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game

    if num_player is None:
        assert len(agents) != 1
        num_player = len(agents)

    runners = [rela.BatchRunner(agent, device, 1000, ["act"]) for agent in agents]
    context = rela.Context()
    games = create_envs(
        num_game,
        seed,
        num_player,
        bomb,
        max_len,
        hand_size=hand_size,
        num_color=num_color,
        num_rank=num_rank,
        num_hint=num_hint,
    )
    threads = []

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []
    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for player_idx in range(num_player):
                idx = player_idx % len(runners)
                actor = hanalearn.R2D2Actor(
                    runners[idx], num_player, player_idx, False, False, False,
                )

                if llm_priors is not None and llm_priors[idx] is not None:
                    assert pikl_lambdas is not None
                    assert pikl_betas is not None
                    actor.set_llm_prior(llm_priors[idx], [pikl_lambdas[idx]], pikl_betas[idx])

                actors.append(actor)
                all_actors.append(actor)
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect, all_actors, games


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    device="cuda:0",
    overwrites=None,
    num_run=1,
    verbose=True,
):
    agents = []
    if overwrites is None:
        overwrites = [{} for _ in range(len(weight_files))]

    llm_priors = [None for _ in weight_files]
    pikl_betas = [1 for _ in weight_files]
    pikl_lambdas = [0 for _ in weight_files]
    for i, weight_file in enumerate(weight_files):
        print("-" * 50)

        agent, cfg = utils.load_agent(weight_file, {"device": device})
        agents.append(agent)

        # if overwrite is not None and cfg[""]:
        if len(overwrites[i]):
            print(f"updating cfg for {weight_file}")
            cfg.update(overwrites[i])

        if "llm_prior" in cfg and isinstance(agent, r2d2.R2D2Agent):
            if cfg["llm_prior"] is not None:
                if overwrites[i] is not None and "llm_prior" in overwrites[i]:
                    llm_pkl = overwrites[i]["llm_prior"]
                    print(f"load pikl prior from {llm_pkl}")
                else:
                    llm_pkl = os.path.join(os.path.dirname(weight_file), "llm.pkl")
                llm_priors[i] = pickle.load(open(llm_pkl, "rb"))
                pikl_betas[i] = cfg["pikl_beta"]
                pikl_lambdas[i] = cfg["pikl_lambda"]
                print(f"lambda {pikl_lambdas[i]}, pikl_beta {pikl_betas[i]}")

        print("-" * 50)
        agent.train(False)

    scores = []
    perfect = 0
    all_games = []
    all_actors = []
    for i in range(num_run):
        _, _, score, p, actors, games = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            device=device,
            llm_priors=llm_priors,
            pikl_betas=pikl_betas,
            pikl_lambdas=pikl_lambdas,
        )
        scores.extend(score)
        perfect += p
        all_games.extend(games)
        all_actors.extend(actors)

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(
            "score: %.3f +/- %.3f" % (mean, sem),
            "; perfect: %.2f%%" % (100 * perfect_rate),
        )
    return mean, sem, perfect_rate, scores, all_actors, all_games


class Evaluator:
    """This version saves some memory, but it is somehow slower"""
    def __init__(
        self,
        agents,
        num_game,
        seed,
        bomb,
        *,
        num_player=None,
        num_thread=10,
        max_len=80,
        device="cuda:0",
        hand_size=5,
        num_color=5,
        num_rank=5,
        num_hint=8,
        llm_prior=None,
        pikl_lambda=0,
        pikl_beta=1,
    ):
        self.agents = agents
        self.num_game = num_game
        self.seed = seed

        if num_game < num_thread:
            num_thread = num_game
        self.num_thread = num_thread

        if num_player is None:
            assert len(agents) > 1
            num_player = len(agents)
        self.runners = [
            rela.BatchRunner(agent, device, 1000, ["act"]) for agent in agents
        ]

        self.context = rela.Context()
        self.games = create_envs(
            num_game,
            seed,
            num_player,
            bomb,
            max_len,
            hand_size=hand_size,
            num_color=num_color,
            num_rank=num_rank,
            num_hint=num_hint,
        )

        assert num_game % num_thread == 0
        game_per_thread = num_game // num_thread
        self.threads = []
        self.all_actors = []
        for t_idx in range(num_thread):
            thread_games = []
            thread_actors = []
            for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
                actors = []
                for i in range(num_player):
                    actor = hanalearn.R2D2Actor(
                        self.runners[i % len(self.agents)], num_player, i, False, False, False,
                    )

                    if llm_prior is not None:
                        actor.set_llm_prior(llm_prior, [pikl_lambda], pikl_beta)

                    actors.append(actor)
                    self.all_actors.append(actor)
                thread_actors.append(actors)
                thread_games.append(self.games[g_idx])
            thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
            self.threads.append(thread)
            self.context.push_thread_loop(thread)

        self.runner_started = False

    def update_agent(self, agents):
        if isinstance(agents, list):
            for i, agent in enumerate(agents):
                self.agents[i].load_state_dict(agent.state_dict())
        else:
            for i in range(len(self.agents)):
                self.agents[i].load_state_dict(agents.state_dict())

    def run(self):
        if not self.runner_started:
            for runner in self.runners:
                runner.start()
            self.runner_started = True

        self.context.start()
        self.context.join()

        for thread in self.threads:
            thread.reset()
        self.context.reset()

        scores = [g.last_episode_score() for g in self.games]
        num_perfect = np.sum([1 for s in scores if s == 25]) / len(scores)
        return np.mean(scores), num_perfect, scores, num_perfect, self.all_actors, self.games

    def clean_up(self):
        for runner in self.runners:
            runner.stop()
