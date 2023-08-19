// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include "rela/transition.h"
#include "rela/utils.h"
#include "rela/batcher.h"

using namespace rela;

RNNTransition::RNNTransition(const RNNTransition& tau, int bsz) {
  isStorage = true;

  obs = allocateBatchStorage(tau.obs, bsz);
  action = allocateBatchStorage(tau.action, bsz);
  h0 = allocateBatchStorage(tau.h0, bsz);
  reward = torch::zeros(getBatchedSize(tau.reward, bsz));
  bootstrap = torch::zeros(getBatchedSize(tau.bootstrap, bsz));
  seqLen = torch::zeros(bsz);
}

void RNNTransition::paste_(const RNNTransition& tau, int idx) {
  assert(isStorage);

  for (auto& kv: tau.obs) {
    obs[kv.first][idx] = kv.second;
  }
  for (auto& kv: tau.action) {
    action[kv.first][idx] = kv.second;
  }
  for (auto& kv: tau.h0) {
    h0[kv.first][idx] = kv.second;
  }
  reward[idx] = tau.reward;
  bootstrap[idx] = tau.bootstrap;
  seqLen[idx] = tau.seqLen;
}

RNNTransition RNNTransition::index(int i) const {
  assert(isStorage);

  RNNTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : h0) {
    element.h0.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.bootstrap = bootstrap[i];
  element.seqLen = seqLen[i];
  return element;
}

void RNNTransition::copyTo(int from, RNNTransition& dst, int to) const {
  assert(isStorage);
  assert(dst.isStorage);

  for (auto& kv : obs) {
    dst.obs[kv.first][to] = kv.second[from];
  }
  for (auto& kv : h0) {
    dst.h0[kv.first][to] = kv.second[from];
  }
  for (auto& kv : action) {
    dst.action[kv.first][to] = kv.second[from];
  }

  dst.reward[to] = reward[from];
  dst.bootstrap[to] = bootstrap[from];
  dst.seqLen[to] = seqLen[from];
}

void RNNTransition::to_(const std::string& device) {
  if (device == "cpu") {
    return;
  }

  auto d = torch::Device(device);
  auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
  obs = tensor_dict::apply(obs, toDevice);
  h0 = tensor_dict::apply(h0, toDevice);
  action = tensor_dict::apply(action, toDevice);
  reward = reward.to(d);
  bootstrap = bootstrap.to(d);
  seqLen = seqLen.to(d);
}

void RNNTransition::seqFirst_() {
  for (auto& kv: obs) {
    obs[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  for (auto& kv: h0) {
    h0[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  for (auto& kv: action) {
    action[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  reward = reward.transpose(0, 1).contiguous();
  bootstrap = bootstrap.transpose(0, 1).contiguous();
  // no need to transpose seqLen
}

RNNTransition rela::makeBatch(
    const std::vector<RNNTransition>& transitions, const std::string& device) {
  std::vector<TensorDict> obsVec;
  std::vector<TensorDict> h0Vec;
  std::vector<TensorDict> actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> bootstrapVec;
  std::vector<torch::Tensor> seqLenVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    obsVec.push_back(transitions[i].obs);
    h0Vec.push_back(transitions[i].h0);
    actionVec.push_back(transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    bootstrapVec.push_back(transitions[i].bootstrap);
    seqLenVec.push_back(transitions[i].seqLen);
  }

  RNNTransition batch;
  batch.obs = tensor_dict::stack(obsVec, 1);
  batch.h0 = tensor_dict::stack(h0Vec, 1);  // 1 is batch for rnn hid
  batch.action = tensor_dict::stack(actionVec, 1);
  batch.reward = torch::stack(rewardVec, 1);
  batch.bootstrap = torch::stack(bootstrapVec, 1);
  batch.seqLen = torch::stack(seqLenVec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
    batch.obs = tensor_dict::apply(batch.obs, toDevice);
    batch.h0 = tensor_dict::apply(batch.h0, toDevice);
    batch.action = tensor_dict::apply(batch.action, toDevice);
    batch.reward = batch.reward.to(d);
    batch.bootstrap = batch.bootstrap.to(d);
    batch.seqLen = batch.seqLen.to(d);
  }

  return batch;
}
