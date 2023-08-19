#pragma once

// #include "cpp/search/game_sim.h"
// #include "cpp/search/player.h"
// #include "rl_search/hand_dist.h"

#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"
#include <torch/extension.h>
#include <vector>

#include "cpp/hanabi_env.h"
#include "cpp/search/player.h"

namespace hle = hanabi_learning_env;

namespace search {

std::vector<std::vector<hle::HanabiCardValue>> filterSample(
    const torch::Tensor& samples,
    const std::vector<int>& privCardCount,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand);

float searchMove(
    const hle::HanabiState& state,
    hle::HanabiMove move,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int>& seeds,
    int myIdx,
    const std::vector<Player>& players);

std::vector<float> parallelSearchMoves(
    const hle::HanabiState& state,
    const std::vector<hle::HanabiMove>& move,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int>& seeds,
    int myIdx,
    const std::vector<Player>& players);

}  // namespace search
