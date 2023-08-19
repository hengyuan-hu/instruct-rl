// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <cmath>
#include <future>
#include <random>
#include <vector>

#include "rela/tensor_dict.h"
#include "rela/transition.h"
#include "rela/concurrent_queue.h"

namespace rela {

template <class DataType>
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta, int prefetch)
      : alpha_(alpha)  // priority exponent
      , beta_(beta)    // importance sampling exponent
      , prefetch_(prefetch)
      , capacity_(capacity)
      , storage_(int(1.25 * capacity))
      , numAdd_(0) {
    assert(prefetch >= 0);
    rng_.seed(seed);
  }

  void clear() {
    assert(sampledIds_.empty());
    while (!futures_.empty()) {
      futures_.pop();
    }
    storage_.clear();
    numAdd_ = 0;
  }

  void resetAlpha(float alpha) {
    alpha_ = alpha;
  }

  void terminate() {
    storage_.terminate();
  }

  void add(const DataType& sample, float priority) {
    numAdd_ += 1;
    storage_.append(sample, std::pow(priority, alpha_));
  }

  void add(const DataType& sample) {
    float priority = 1.0;
    add(sample, priority);
  }

  std::tuple<DataType, torch::Tensor> sample(int batchsize, const std::string& device) {
    if (!sampledIds_.empty()) {
      std::cout << "Error: previous samples' priority has not been updated." << std::endl;
      assert(false);
    }

    DataType batch;
    torch::Tensor priority;
    if (prefetch_ == 0) {
      std::tie(batch, priority, sampledIds_) = sample_(batchsize, device);
      return std::make_tuple(batch, priority);
    }

    if (futures_.empty()) {
      std::tie(batch, priority, sampledIds_) = sample_(batchsize, device);
    } else {
      std::tie(batch, priority, sampledIds_) = futures_.front().get();
      futures_.pop();
    }

    while ((int)futures_.size() < prefetch_) {
      auto f = std::async(
          std::launch::async,
          &PrioritizedReplay<DataType>::sample_,
          this,
          batchsize,
          device);
      futures_.push(std::move(f));
    }

    return std::make_tuple(batch, priority);
  }

  void updatePriority(const torch::Tensor& priority) {
    if (priority.size(0) == 0) {
      sampledIds_.clear();
      return;
    }

    assert(priority.dim() == 1);
    assert((int)sampledIds_.size() == priority.size(0));

    auto weights = torch::pow(priority, alpha_);
    {
      std::lock_guard<std::mutex> lk(mSampler_);
      storage_.update(sampledIds_, weights);
    }
    sampledIds_.clear();
  }

  DataType getFirstK(int size, const std::string& device) {
    std::vector<DataType> samples;
    for (int i = 0; i < size; ++i) {
      samples.push_back(storage_.get(i));
    }
    return makeBatch(samples, device);;
  }

  DataType get(int idx) {
    return storage_.get(idx);
  }

  DataType getRange(int start, int end, const std::string& device) {
    std::vector<DataType> samples;
    for (int i = start; i < end; ++i) {
      samples.push_back(storage_.get(i));
    };
    return makeBatch(samples, device);
  }

  int size() const {
    return storage_.safeSize(nullptr);
  }

  int numAdd() const {
    return numAdd_;
  }

 private:
  using SampleWeightIds = std::tuple<DataType, torch::Tensor, std::vector<int>>;

  SampleWeightIds sample_(int batchsize, const std::string& device) {
    std::unique_lock<std::mutex> lk(mSampler_);

    float sum;
    int size = storage_.safeSize(&sum);
    assert(size >= batchsize);
    // storage_ [0, size) remains static in the subsequent section

    float segment = sum / batchsize;
    std::uniform_real_distribution<float> dist(0.0, segment);

    std::vector<DataType> samples;
    auto weights = torch::zeros({batchsize}, torch::kFloat32);
    auto weightAcc = weights.accessor<float, 1>();
    std::vector<int> ids(batchsize);

    double accSum = 0;
    int nextIdx = 0;
    float w = 0;
    int id = 0;
    for (int i = 0; i < batchsize; i++) {
      float rand = dist(rng_) + i * segment;
      rand = std::min(sum - (float)0.1, rand);

      while (nextIdx <= size) {
        if (accSum > 0 && accSum >= rand) {
          assert(nextIdx >= 1);
          DataType element = storage_.getElementAndMark(nextIdx - 1);
          samples.push_back(element);
          weightAcc[i] = w;
          ids[i] = id;
          break;
        }

        if (nextIdx == size) {
          std::cout << "nextIdx: " << nextIdx << "/" << size << std::endl;
          std::cout << std::setprecision(10) << "accSum: " << accSum << ", sum: " << sum
                    << ", rand: " << rand << std::endl;
          assert(false);
        }

        w = storage_.getWeight(nextIdx, &id);
        accSum += w;
        ++nextIdx;
      }
    }
    assert((int)samples.size() == batchsize);

    // pop storage if full
    size = storage_.size();
    if (size > capacity_) {
      storage_.blockPop(size - capacity_);
    }

    // safe to unlock, because <samples> contains copys
    lk.unlock();

    weights = weights / sum;
    weights = torch::pow(size * weights, -beta_);
    weights /= weights.max().detach();
    if (device != "cpu") {
      weights = weights.to(torch::Device(device));
    }
    auto batch = makeBatch(samples, device);
    return std::make_tuple(batch, weights, ids);
  }

  float alpha_;
  const float beta_;
  const int prefetch_;
  const int capacity_;

  ConcurrentQueue<DataType> storage_;
  std::atomic<int> numAdd_;

  // make sure that sample & update does not overlap
  std::mutex mSampler_;
  std::vector<int> sampledIds_;
  std::queue<std::future<SampleWeightIds>> futures_;

  std::mt19937 rng_;
};

using RNNPrioritizedReplay = PrioritizedReplay<RNNTransition>;

}  // namespace rela
