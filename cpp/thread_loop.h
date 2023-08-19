#pragma once

#include "cpp/r2d2_actor.h"
#include "rela/thread_loop.h"

class HanabiThreadLoop : public rela::ThreadLoop {
 public:
  HanabiThreadLoop(
      std::vector<std::shared_ptr<HanabiEnv>> envs,
      std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors,
      bool eval)
      : envs_(std::move(envs))
      , actors_(std::move(actors))
      , done_(envs_.size(), -1)
      , eval_(eval) {
    assert(envs_.size() == actors_.size());
  }

  virtual void mainLoop() override;

  void reset() {
    assert(eval_); // reset is only for the eval mode
    // set done_ = 0 instead of -1 because the game already starts after reset
    std::fill(done_.begin(), done_.end(), 0);
    numDone_ = 0;
    for (size_t i = 0; i < envs_.size(); ++i) {
      envs_[i]->reset();
      for (size_t j = 0; j < actors_[i].size(); ++j) {
          actors_[i][j]->reset(*envs_[i]);
      }
    }
  }

 private:
  std::vector<std::shared_ptr<HanabiEnv>> envs_;
  std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors_;
  std::vector<int8_t> done_;
  const bool eval_;
  int numDone_ = 0;
};
