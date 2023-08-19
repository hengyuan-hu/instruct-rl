from collections import defaultdict
from typing import Dict, Sequence, Tuple, List, Any
import random
import copy
import itertools
import numpy as np

from ball_env import BallEnv
from llm import gpt2_predict, gpt2_predict_fake
from optim import AdamOptimizer


class QAgent:
    def __init__(self, player_idx, num_action, replay_size):
        self._player_idx = player_idx
        self._num_action = num_action
        self._replay_size = replay_size
        self._action_names = {i: str(i + 1) for i in range(self._num_action)}
        if player_idx == 1:
            self._action_names[self._num_action - 1] = "0"
        self._action_names[-1] = "no-op"
        # print(f"player {self._player_idx}")
        # print(f"action_names: {self._action_names}")

        self.q_table = {}
        self.experience = []
        self.current_epsd = []
        self.verbose = False

    def reset(self):
        self.current_epsd = []

    def clone(self):
        raise NotImplementedError

    def get_q(self, obs):
        if obs in self.q_table:
            return self.q_table[obs]

        _, active_player, _ = obs
        if active_player == self._player_idx:
            q_values = {
                i: np.random.uniform(-0.01, 0.01) for i in range(self._num_action)
            }
        else:
            q_values = {-1: np.random.uniform(-0.01, 0.01)}

        self.q_table[obs] = q_values
        return q_values

    def decide_action(self, env: BallEnv, eps: float) -> int:
        obs = env.observe(self._player_idx)
        if np.random.uniform(0, 1) < eps:
            # random
            q_values = self.get_q(obs)
            action = random.choice(list(q_values.keys()))
        else:
            action, _ = self._greedy_act(obs)

        # if self.verbose:
        #     obs = env.observe(self._player_idx)
        #     q_values = self.get_q(obs)
        #     print(f"player {self._player_idx}")
        #     print(f"state: {obs}")
        #     for a, q in q_values.items():
        #             print(f"action={a}: q-value={q:.3f}")
        #     print(f"action: {self._action_names[action]}")
        #     print(f"----------------")

        return action

    def _greedy_act(self, obs) -> Tuple[int, float]:
        q_values = self.get_q(obs)

        best_action = -100
        best_q = -100
        for action, q_value in q_values.items():
            if q_value > best_q:
                best_action = action
                best_q = q_value
        assert best_action != -100
        return best_action, q_values[best_action]

    def collect_experience(
        self,
        obs,
        action: int,
        reward: int,
        terminal: bool,
        next_obs,
    ):
        self.current_epsd.append((obs, action, reward, terminal, next_obs))

    def end_of_episode(self):
        if len(self.current_epsd) == 0:
            # no data has been collected
            return

        self.experience.append(self.current_epsd)
        if len(self.experience) > self._replay_size:
            self.experience.pop(0)

    def learn(self, lr: float, gamma: float, batchsize: int):
        assert batchsize == 1, "kidding me?"
        if len(self.experience) <= 0.1 * self._replay_size:
            return

        epsd = random.choice(self.experience)
        for obs, action, reward, terminal, next_obs in epsd:
            q_target = 0
            if not terminal:
                _, q_target = self._greedy_act(next_obs)
            target = gamma * q_target + reward
            qa = self.get_q(obs)[action]  # init if not exist
            # print(f">>>err: {target - qa}")
            self.q_table[obs][action] += lr * (target - qa)


class LangQAgent(QAgent):
    def __init__(
        self,
        player_idx,
        num_action,
        replay_size,
        prompt,
        lambda_,
        model,
        tokenizer,
        cache,
        use_fake_pred=False,
        log_q_history=False,
        use_adam=False,
        adam_eps=None,
    ):
        super().__init__(player_idx, num_action, replay_size)
        self._prompt = prompt
        self._lambda = lambda_
        self._model = model
        self._tokenizer = tokenizer
        self._cache = cache
        self._vdn = False
        self._partner = None
        self._use_fake_pred = use_fake_pred
        self._log_q_history = log_q_history
        self._q_history = []
        self._obs_action_count = {}
        self._obs_action_update_count = {}
        self._optim = None
        if use_adam:
            if adam_eps is not None:
                self._optim = AdamOptimizer(eps=adam_eps)
            else:
                self._optim = AdamOptimizer()

    def set_partner(self, partner):
        self._vdn = True
        self._partner = partner

    def clone(self):
        cloned = LangQAgent(
            self._player_idx,
            self._num_action,
            self._replay_size,
            self._prompt,
            self._lambda,
            self._model,
            self._tokenizer,
            self._cache,
            use_fake_pred=self._use_fake_pred,
        )
        cloned.q_table = copy.deepcopy(self.q_table)
        if self._partner is not None:
            cloned.set_partner(self._partner)
        return cloned

    def get_policy(self):
        obs_actions = []
        for obs in self.q_table:
            _, active_player, _ = obs
            if active_player != self._player_idx:
                continue

            action, _ = self._greedy_act(obs)
            obs_actions.append((obs, action))
        return obs_actions

    def decide_action(self, env: BallEnv, eps: float) -> int:
        # action = super().decide_action(env, eps)
        obs = env.observe(self._player_idx)
        if np.random.uniform(0, 1) < eps:
            # random
            q_values = self.get_q(obs)
            action = random.choice(list(q_values.keys()))
        else:
            action, _ = self._greedy_act(obs)

        if self._log_q_history:
            obs = env.observe(self._player_idx)
            if (obs, action) not in self._obs_action_count:
                self._obs_action_count[(obs, action)] = 0
            self._obs_action_count[(obs, action)] += 1
        return action

    def _greedy_act(self, obs) -> Tuple[int, float]:
        _, active_player, past_action = obs

        q_values = self.get_q(obs)
        if active_player != self._player_idx:
            # not my turn
            assert len(q_values) == 1
            action, q_value = list(q_values.items())[0]
            return action, q_value

        if self.verbose:
            if isinstance(past_action, Tuple):
                print(
                    f"player_idx {self._player_idx}, "
                    f"active_player {active_player}, "
                    f"prev action: {[self._action_names[a] for a in past_action]}"
                )
            else:
                print(
                    f"player_idx {self._player_idx}, "
                    f"active_player {active_player}, "
                    f"prev action: {[self._action_names[past_action]]}"
                )

        if past_action != ():
            if isinstance(past_action, Tuple):
                past_action = past_action[-1]
            prompt = (
                f"{self._prompt} My partner selected {self._action_names[past_action]}."
            )
        else:
            prompt = self._prompt

        prompt = f"{prompt} So I should select"
        predictor = gpt2_predict_fake if self._use_fake_pred else gpt2_predict
        logps = predictor(
            self._model,
            self._tokenizer,
            prompt,
            [f" {self._action_names[i]}" for i in range(self._num_action)],
            self._cache,
        )

        if self.verbose:
            print(f"state: {obs}")
            print(f"prompt: {prompt}")
            for action, q_value in q_values.items():
                print(
                    f"{self._action_names[action]}: "
                    f"q-value = {q_value:6.3f}, p = {np.exp(logps[action]):6.3f}, logp = {logps[action]:6.3f}, "
                    f"combined = {q_value + self._lambda * logps[action]:6.3f}"
                )

        best_action = -100
        best_value = -100
        for action, q_value in q_values.items():
            combined_value = q_value + self._lambda * logps[action]
            if combined_value > best_value:
                best_value = combined_value
                best_action = action

        # assert best_action >= 0
        if best_action < 0:
            print(q_values.items())
            assert False

        if self.verbose:
            print(f"action: {self._action_names[best_action]}")
            print(f"----------------")
        return best_action, q_values[best_action]

    def learn(self, lr: float, gamma: float, batchsize):
        if not self._vdn:
            return super().learn(lr, gamma, batchsize)

        if self._player_idx != 0:
            # player 0 will perform learning for everyone
            return

        if len(self.experience) <= min(batchsize, 0.1 * self._replay_size):
            return

        grads = [defaultdict(list) for _ in range(2)]  # two players
        for _ in range(batchsize):
            self._compute_grad(gamma, grads)

        # avg_grads = {}
        assert self._partner is not None  # to make type checking happy
        for aid, agent in enumerate([self, self._partner]):
            agent_grads = grads[aid]
            for (obs, action), gs in agent_grads.items():
                grad = np.mean(gs)
                if self._optim is not None:
                    grad = self._optim.process_grad(obs, action, grad)
                agent.q_table[obs][action] += lr * grad

            if agent._log_q_history:
                agent._q_history.append(copy.deepcopy(agent.q_table))

    def _compute_grad(self, gamma, grads: Sequence[Dict[Tuple[Any, ...], List[float]]]):
        assert self._partner is not None
        assert len(self.experience) == len(self._partner.experience)

        idx = np.random.randint(0, len(self.experience))
        q_targets: List[float] = []
        rewards: List[float] = []
        agent_qs = []
        for aid, agent in enumerate([self, self._partner]):
            agent_qs.append([])

            epsd = agent.experience[idx]
            for t, (obs, action, reward, terminal, next_obs) in enumerate(epsd):
                agent_qs[-1].append(agent.get_q(obs)[action])
                if aid == 0:
                    rewards.append(reward)
                else:
                    assert rewards[t] == reward

                q_target = 0
                if not terminal:
                    _, q_target = agent._greedy_act(next_obs)

                if aid == 0:
                    q_targets.append(q_target)
                else:
                    q_targets[t] += q_target

        # assert shared_reward is not None
        # target = gamma * sum_q_target + shared_reward
        for aid, agent in enumerate([self, self._partner]):
            epsd = agent.experience[idx]
            for t, (obs, action, _, _, _) in enumerate(epsd):
                # obs, action, _, _, _ = epsd
                qa = agent.get_q(obs)[action]  # init if not exist
                partner_qa = agent_qs[1 - aid][t]
                target = q_targets[t] * gamma + rewards[t]
                grad = target - (qa + partner_qa)
                grads[aid][(obs, action)].append(grad)
        return


def run(
    env: BallEnv,
    agents: Sequence[QAgent],
    num_game,
    lr,
    gamma,
    eps,
    is_train,
    print_intv,
    batchsize,
):
    total_scores = []
    for i in range(num_game):
        env.reset()
        for agent in agents:
            agent.reset()

        while not env.is_terminal():
            if agents[0].verbose:
                print(f"================step: {env.remaining_step}================")
            actions = []
            experiences = []
            for player_idx, agent in enumerate(agents):
                action = agent.decide_action(env, eps)
                actions.append(action)
                if is_train:
                    experiences.append([env.observe(agent._player_idx), action])

            action = actions[env.active_player]
            reward = env.apply_action(action)
            terminal = env.is_terminal()
            if agents[0].verbose:
                print(f"reward = {reward}")

            if is_train:
                for player_idx, agent in enumerate(agents):
                    experiences[player_idx].append(reward)
                    experiences[player_idx].append(terminal)
                    experiences[player_idx].append(env.observe(player_idx))
                    agent.collect_experience(*experiences[player_idx])

        assert env.is_terminal()
        if agents[0].verbose:
            print(f"<<<<<<< score: {env.score_percentage} >>>>>>>>")

        total_scores.append(env.score_percentage)
        for agent in agents:
            agent.end_of_episode()

        for agent in agents:
            if is_train:
                # if i == num_game / 2:
                #     lr = lr / 2
                lrr = lr  # * (1 - i / num_iter)
                agent.learn(lrr, gamma, batchsize)

        if print_intv > 0 and (i + 1) % print_intv == 0:
            mean_total_score = (
                np.mean(total_scores[-print_intv:]) if len(total_scores) else 0
            )
            print(f"iter-{i+1}, total scores: {mean_total_score:.2f}")

    return agents, float(np.mean(total_scores))


def eval_agents(
    env: BallEnv,
    agents: Sequence[QAgent],
):
    all_reward_balls = []
    for k in range(1, env._max_num_reward + 1):
        n_choose_k = itertools.combinations(list(range(env._num_ball)), k)
        possible_reward_balls = list(n_choose_k)
        all_reward_balls.extend(possible_reward_balls)
    # print(f"# possible envs: {len(all_reward_balls)}")

    total_scores = []
    for reward_balls in all_reward_balls:
        ball_rewards = [-1 for _ in range(env._num_ball)]
        for b in reward_balls:
            ball_rewards[b] = 1

        env.reset(ball_rewards)
        for agent in agents:
            agent.reset()

        # # if agents[0].verbose:
        #     print("\n\n")
        while not env.is_terminal():
            if agents[0].verbose:
                print(f"================step: {env.remaining_step}================")
            actions = []
            # experiences = []
            for agent in agents:
                action = agent.decide_action(env, 0)
                actions.append(action)

            # print(">>", env.active_player, len(actions))
            action = actions[env.active_player]
            reward = env.apply_action(action)
            # terminal = env.is_terminal()
            if agents[0].verbose:
                print(f"reward = {reward}")

        assert env.is_terminal()
        if agents[0].verbose:
            print(f"<<<<<<< score: {env.score_percentage} >>>>>>>>")

        total_scores.append(env.score_percentage)
        # env.reset()
        for agent in agents:
            agent.end_of_episode()

    # print(">>>", len(agents))
    return float(np.mean(total_scores))


def eval_agents_specific(
    env: BallEnv,
    agents: Sequence[QAgent],
    ball_rewards: list[int],
):
    env.reset(ball_rewards)
    for agent in agents:
        agent.reset()

    while not env.is_terminal():
        if agents[0].verbose:
            print(f"================step: {env.remaining_step}================")
        actions = []
        for agent in agents:
            action = agent.decide_action(env, 0)
            actions.append(action)

        action = actions[env.active_player]
        reward = env.apply_action(action)
        if agents[0].verbose:
            print(f"reward = {reward}")

    assert env.is_terminal()
    if agents[0].verbose:
        print(f"<<<<<<< score: {env.score_percentage} >>>>>>>>")

    for agent in agents:
        agent.end_of_episode()


def show_agent_conventions(agent):
    assert agent._player_idx == 1
    obs_actions = agent.get_policy()

    def sort_func(obs_action):
        obs, _ = obs_action
        past_action = obs[-1]
        if len(past_action) == 1:
            return past_action[0] - 100
        return past_action[1] * 10 + past_action[0]

    obs_actions = sorted(obs_actions, key=sort_func)
    last_actions = None
    is_human_policy = True
    for obs, action in obs_actions:
        if last_actions is not None:
            if len(obs[2]) > len(last_actions):
                print("-" * 20)
            elif obs[2][-1] != last_actions[-1] and len(obs[2]) > 1:
                print("-" * 20)
        if action == 5:
            print(f"{[agent._action_names[a] for a in obs[2]]} -> quit")
            if len(obs[2]) != 2 or obs[2][0] != obs[2][1]:
                is_human_policy = False
        else:
            print(
                f"{[agent._action_names[a] for a in obs[2]]} -> {agent._action_names[action]}"
            )
            if obs[2][-1] != action:
                is_human_policy = False

        last_actions = obs[2]
    return is_human_policy


def show_agent_conventions_simple_env(agent, verbose=False):
    if verbose:
        agent.verbose = True
    assert agent._player_idx == 1
    obs_actions = agent.get_policy()

    obs_actions = sorted(obs_actions, key=lambda x: x[0][-1])
    for obs, action in obs_actions:
        if action == 5:
            print(f"{[agent._action_names[obs[2]]]} -> quit")
        else:
            print(f"{[agent._action_names[obs[2]]]} -> {agent._action_names[action]}")

    if verbose:
        agent.verbose = False
