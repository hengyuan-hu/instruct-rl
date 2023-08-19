#include "thread_loop.h"

void HanabiThreadLoop::mainLoop() {
  while (!terminated()) {
    // go over each envs in sequential order
    // call in seperate for-loops to maximize parallization
    for (size_t i = 0; i < envs_.size(); ++i) {
      if (paused()) {
        waitUntilResume();
      }

      if (done_[i] == 1) {
        continue;
      }

      assert(done_[i] < 1);
      auto& actors = actors_[i];

      // only check if game terminates and whether to
      // reset the game & actor if the all actor has
      // finished their tasks with the current game step
      bool allActorDone = true;
      for (auto& actor : actors) {
        if (!actor->stepDone()) {
          allActorDone = false;
        }
      }
      if (allActorDone && envs_[i]->terminated()) {
        if (eval_) {
          // we only run 1 game for evaluation. done[i] is initially
          // -1, and become 0 when we start the first game.
          ++done_[i];
          if (done_[i] == 1) {
            numDone_ += 1;
            if (numDone_ == (int)envs_.size()) {
              return;
            } else {
              continue;
            }
          }
        }

        envs_[i]->reset();
        for (size_t j = 0; j < actors.size(); ++j) {
          actors[j]->reset(*envs_[i]);
        }
      }

      bool allActorReady = true;
      for (auto& actor : actors) {
        if (!actor->ready()) {
          allActorReady = false;
        }
      }
      if (!allActorReady) {
        // we want to keep sync between actors in a same game
        continue;
      }

      std::vector<std::unique_ptr<hle::HanabiMove>> moves;
      for (auto& actor : actors) {
        moves.push_back(actor->next(*envs_[i]));
      }
      if (!envs_[i]->terminated()) {
        // std::cout << "cur player?: " << envs_[i]->getCurrentPlayer() << std::endl;
        auto& move = moves[envs_[i]->getCurrentPlayer()];
        if (move != nullptr) {
          // std::cout << "move: " << move->ToString() << std::endl;
          envs_[i]->step(*move);
        }
      }
    }
  }
}
