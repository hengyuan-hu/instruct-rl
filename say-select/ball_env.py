from typing import List
import numpy as np


class BallEnv:
    def __init__(self, num_ball: int, max_num_reward: int, num_player: int):
        # constants
        assert max_num_reward <= num_ball
        self._num_ball = num_ball
        self._max_num_reward = max_num_reward
        self._num_step = 2 * num_ball
        self._num_player = num_player

        # game related
        self.remaining_step = 0
        self.active_player = 0
        self.ball_rewards = []
        self.past_actions: List[int] = []
        self.total_score = 0
        self.max_score = 0

    def __repr__(self) -> str:
        return f"balls: {self.ball_rewards}, remaining_steps: {self.remaining_step}"

    def reset(self, ball_rewards=None):
        self.remaining_step = self._num_step
        self.active_player = 0
        self.past_actions = []
        self.total_score = 0

        if ball_rewards is not None:
            assert len(ball_rewards) == self._num_ball
            assert ball_rewards.count(1) <= self._max_num_reward
            assert ball_rewards.count(1) + ball_rewards.count(-1) == len(ball_rewards)
            self.max_score = ball_rewards.count(1)
            self.ball_rewards = ball_rewards
        else:
            # num_neg_reward = np.random.choice(self._num_reward, size=1)[0]
            num_pos_reward = np.random.randint(1, self._max_num_reward + 1)
            self.max_score = num_pos_reward
            assert self.max_score > 0
            reward_balls = np.random.choice(
                self._num_ball, size=num_pos_reward, replace=False
            )
            self.ball_rewards = [-1 for _ in range(self._num_ball)]
            for b in reward_balls:
                self.ball_rewards[b] = 1

    def observe(self, player_idx):
        """
        The first player observes the location of the ball & past action.
        The second player observes the action of the first player.
        """
        state = self.ball_rewards.copy()
        past_action = self.past_actions[-1] if len(self.past_actions) else ()
        past_actions = tuple(self.past_actions[-2:])
        if player_idx == 0:
            return tuple(state), self.active_player, past_action
        else:
            return tuple(), self.active_player, past_actions

    def apply_action(self, action: int):
        """
        action:
            active_player = 0:
                [0, num_ball-1] -> indicating different balls.
            active_player = 1:
                action in [0, num_ball-1] -> select the ball
                action = num_ball -> terminate the game.
        """
        assert self.remaining_step > 0
        assert action >= 0
        self.remaining_step -= 1

        self.past_actions.append(action)
        reward: int = 0
        if self.active_player == 1:
            if action == len(self.ball_rewards):
                # quit
                self.remaining_step = 0
                return 0

            reward = self.ball_rewards[action]
            self.ball_rewards[action] = -1  # reset to -1
            self.total_score += reward

        self.active_player = (self.active_player + 1) % self._num_player
        return reward

    def is_terminal(self):
        if self.remaining_step == 0:
            return True
        for r in self.ball_rewards:
            if r != 0:
                return False
        return True

    @property
    def score_percentage(self):
        return self.total_score / self.max_score
