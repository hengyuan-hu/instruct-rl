// common components for implementing R2D2 actors
#include "rela/r2d2.h"
#include "rela/batcher.h"

using namespace rela;

  // TensorDict& obsBack() {
  //   if (callOrder_ == 0) {
  //     assert(seqLen_ > 0);
  //     return obs_[seqLen_ - 1];
  //   } else {
  //     return obs_[seqLen_];
  //   }
  // }

  // TensorDict& obs(int i) {
  //   assert(i < seqLen_);
  //   return obs_[i];
  // }

  void R2D2Buffer::pushObs(const TensorDict& obs) {
    assert(callOrder_ == 0);
    ++callOrder_;
    assert(seqLen_ < maxSeqLen_);

    if (transition_.obs.size() == 0) {
      transition_.obs = allocateBatchStorage(obs, maxSeqLen_);
    }
    for (auto& kv : obs) {
      transition_.obs[kv.first][seqLen_] = kv.second;
    }
  }

  void R2D2Buffer::pushAction(const TensorDict& action) {
    assert(callOrder_ == 1);
    ++callOrder_;

    if (transition_.action.size() == 0) {
      transition_.action = allocateBatchStorage(action, maxSeqLen_);
    }
    for (auto& kv: action) {
      transition_.action[kv.first][seqLen_] = kv.second;
    }
  }

  void R2D2Buffer::pushReward(float r) {
    assert(callOrder_ == 2);
    ++callOrder_;
    reward_[seqLen_] = r;
  }

  void R2D2Buffer::pushTerminal(float t) {
    assert(callOrder_ == 3);
    callOrder_ = 0;
    terminal_[seqLen_] = t;
    ++seqLen_;
  }

  void R2D2Buffer::reset() {
    assert(callOrder_ == 0);
    assert(terminal_[seqLen_ - 1] == 1.0f);
    seqLen_ = 0;
    callOrder_ = 0;
  }

  RNNTransition R2D2Buffer::popTransition() {
    assert(callOrder_ == 0);
    // episode has to terminate
    assert(terminal_[seqLen_ - 1] == 1.0f);

    // auto len = transition_.seqLen.accessor<float, 0>();
    // len = float(seqLen_);
    transition_.seqLen.data_ptr<float>()[0] = float(seqLen_);
    auto accReward = transition_.reward.accessor<float, 1>();
    auto bootstrap = transition_.bootstrap.accessor<float, 1>();
    // acc reward
    for (int i = 0; i < seqLen_; ++i) {
      float factor = 1;
      float acc = 0;
      for (int j = 0; j < multiStep_; ++j) {
        if (i + j >= seqLen_) {
          break;
        }
        acc += factor * reward_[i + j];
        factor *= gamma_;
      }
      accReward[i] = acc;
    }

    for (int i = 0; i < seqLen_; ++i) {
      if (i < seqLen_ - multiStep_) {
        bootstrap[i] = 1.0f;
      } else {
        bootstrap[i] = 0.0f;
      }
    }

    // std::cout << "seqLen: " << seqLen_ << ", seqLen(tensor):" << transition_.seqLen << std::endl;
    // std::cout << "acc reward: " << std::endl;
    // for (int i = 0; i < maxSeqLen_; ++i) {
    //   std::cout << transition_.reward[i].item<float>();
    //   if (i == seqLen_-1) {
    //     std::cout << "(X)";
    //   }
    //   std::cout << ", ";
    //   if ((i + 1) % 10 == 0) {
    //     std::cout << std::endl;
    //   }
    // }

    // std::cout << "bootstrap: " << std::endl;
    // for (int i = 0; i < maxSeqLen_; ++i) {
    //   std::cout << transition_.bootstrap[i].item<float>();
    //   if (i == seqLen_-1) {
    //     std::cout << "(X)";
    //   }
    //   std::cout << ", ";
    //   if ((i + 1) % 10 == 0) {
    //     std::cout << std::endl;
    //   }
    // }

    // // padding
    // for (int i = seqLen_; i < maxSeqLen_; ++i) {
    //   obs_[i] = tensor_dict::zerosLike(obs_[seqLen_ - 1]);
    //   action_[i] = tensor_dict::zerosLike(action_[seqLen_ - 1]);
    //   reward_[i] = 0.f;
    //   terminal_[i] = 1.0f;
    //   accReward_[i] = 0.0f;
    // }

    // RNNTransition transition;
    // transition.obs = tensor_dict::stack(obs_, 0);
    // transition.action = tensor_dict::stack(action_, 0);
    // transition.reward = torch::tensor(accReward_);
    // transition.bootstrap = torch::tensor(bootstrap_);
    // transition.seqLen = torch::tensor(float(seqLen_));
    // transition.h0 = h0_;

    seqLen_ = 0;
    callOrder_ = 0;
    // return transition;
    return transition_;
  }

