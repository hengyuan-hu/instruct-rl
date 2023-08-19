#pragma once

#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rela/batch_runner.h"
#include "rela/tensor_dict.h"

#include "cpp/utils.h"

namespace hle = hanabi_learning_env;

namespace search {

struct SimHand {
  int index;
  std::vector<hle::HanabiCardValue> cards;

  SimHand(int index, const std::vector<hle::HanabiCardValue>& cards)
      : index(index)
      , cards(cards) {
  }
};

class GameSimulator {
 public:
  GameSimulator(
      const hle::HanabiState& refState, const std::vector<SimHand>& simHands, int newSeed)
      : game_(*refState.ParentGame())
      , state_(refState) {
    game_.SetSeed(newSeed);
    reset(refState, simHands);
  }

  GameSimulator(const GameSimulator& sim)
      : game_(sim.game_)
      , state_(sim.state_) {
    state_.SetGame(&game_);
  };

  GameSimulator& operator=(const GameSimulator&) = delete;
  GameSimulator(GameSimulator&&) = delete;
  GameSimulator& operator=(GameSimulator&&) = delete;

  void reset(const hle::HanabiState& refState, const std::vector<SimHand>& simHands);

  void step(hle::HanabiMove move) {
    std::tie(reward_, terminal_) = applyMove(state_, move, false);
  }

  hle::HanabiMove getMove(int uid) const {
    return game_.GetMove(uid);
  }

  const hle::HanabiState& state() const {
    return state_;
  }

  const hle::HanabiGame& game() const {
    return game_;
  }

  float reward() const {
    return reward_;
  }

  bool terminal() const {
    return terminal_;
  }

  void setTerminal(bool terminal) {
    terminal_ = terminal;
  }

  int score() const {
    return state_.Score();
  }

 private:
  hle::HanabiGame game_;
  hle::HanabiState state_;

  bool terminal_ = false;
  float reward_ = 0;
};
}  // namespace search
