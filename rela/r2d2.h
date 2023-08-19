// common components for implementing R2D2 actors
#pragma once

#include "rela/tensor_dict.h"
#include "rela/transition.h"

namespace rela {

class R2D2Buffer {
 public:
  R2D2Buffer(int multiStep, int maxSeqLen, float gamma)
      : multiStep_(multiStep)
      , maxSeqLen_(maxSeqLen)
      , gamma_(gamma)
      , callOrder_(0)
      , seqLen_(0)
      , reward_(maxSeqLen)
      , terminal_(maxSeqLen) {
      assert(maxSeqLen > 0);
      transition_.reward = torch::zeros(maxSeqLen, torch::kFloat32);
      transition_.bootstrap = torch::zeros(maxSeqLen, torch::kFloat32);
      transition_.seqLen = torch::tensor(float(0));
  }

  void init(const TensorDict& h0) {
    // h0_ = h0;
    transition_.h0 = h0;
  }

  int len() const {
    return seqLen_;
  }

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

  void pushObs(const TensorDict& obs);

  void pushAction(const TensorDict& action);

  void pushReward(float r);

  void pushTerminal(float t);

  void reset();

  RNNTransition popTransition();

 private:
  const int multiStep_;
  const int maxSeqLen_;
  const float gamma_;

  // TensorDict h0_;
  // std::vector<TensorDict> obs_;
  // std::vector<TensorDict> action_;
  // torch::Tensor reward_;
  // torch::Tensor terminal_;
  // torch::Tensor bootstrap_;
  // torch::Tensor accReward_;
  // torch::Tensor seqLen_;
  RNNTransition transition_;

  int callOrder_;
  int seqLen_;
  std::vector<float> reward_;
  std::vector<float> terminal_;

  // // derived
  // std::vector<float> bootstrap_;
  // std::vector<float> accReward_;

};

}  // namespace rela
