// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <torch/extension.h>

#include "tensor_dict.h"

namespace rela {

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition(const RNNTransition&, int bsz);

  void paste_(const RNNTransition&, int idx);

  RNNTransition index(int i) const;

  void copyTo(int from, RNNTransition& dst, int to) const;

  void to_(const std::string& device);

  void seqFirst_();

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;

  bool isStorage = false;
};


RNNTransition makeBatch(
    const std::vector<RNNTransition>& transitions, const std::string& device);

}  // namespace rela
