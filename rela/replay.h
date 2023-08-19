#pragma once

#include <random>
#include <thread>
#include <vector>
#include <queue>

#include "rela/tensor_dict.h"
#include "rela/transition.h"
#include "rela/concurrent_queue.h"

namespace rela {

class Replay {
 public:
  Replay(int capacity, int seed, int prefetch)
      : prefetch_(prefetch)
      , capacity_(capacity)
      , storage_(int(1.25 * capacity))
      , numAdd_(0) {
    rng_.seed(seed);
  }

  void clear() {
    assert(false); // not yet checked after switching to thread

    storage_.clear();
    numAdd_ = 0;
  }

  void terminate() {
    storage_.terminate();
  }

  void add(const RNNTransition& sample) {
    numAdd_ += 1;
    storage_.append(sample, 1);
  }

  RNNTransition sample(int batchsize, const std::string& device) {
    // simple, single thread version
    if (prefetch_ == 0) {
      return sample_(batchsize, device);
    }

    if (samplerThread_ == nullptr) {
      // create sampler thread
      samplerThread_ = std::make_unique<std::thread>(
          &Replay::sampleLoop_, this, batchsize, device);
    }

    // std::cout << "Try to get batch" << std::endl;
    std::unique_lock<std::mutex> lk(mSampler_);
    cvSampler_.wait(lk, [this] {return samples_.size() > 0;});
    // std::cout << "Get Batch" << std::endl;

    auto batch = samples_.front();
    samples_.pop();

    lk.unlock();
    cvSampler_.notify_all();
    // std::cout << "Return Batch" << std::endl;
    return batch;
  }

  RNNTransition get(int idx) {
    return storage_.get(idx);
  }

  RNNTransition getRange(int start, int end, const std::string& device) {
    std::vector<RNNTransition> samples;
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
  void sampleLoop_(int batchsize, const std::string& device) {
    while (true) {
      auto batch = sample_(batchsize, device);

      std::unique_lock<std::mutex> lk(mSampler_);
      cvSampler_.wait(lk, [this] {return (int)samples_.size() < prefetch_;});
      samples_.push(batch);
      // std::cout << "samples size: " << samples_.size() << std::endl;
      lk.unlock();
      cvSampler_.notify_all();
    }
  }

  RNNTransition sample_(int batchsize, const std::string& device) {
    float sum;
    int size = storage_.safeSize(&sum);
    assert(int(sum) == size);
    assert(size >= batchsize);
    // storage_ [0, size) remains static in the subsequent section
    int segment = size / batchsize;
    std::uniform_int_distribution<int> dist(0, segment-1);

    assert(batchsize > 0);
    RNNTransition batch(storage_.get(0), batchsize);

    // RNNTransition batch;
    for (int i = 0; i < batchsize; ++i) {
      int rand = dist(rng_) + i * segment;
      assert(rand < size);
      storage_.copyTo(rand, batch, i);
    }

    // pop storage if full
    size = storage_.size();
    if (size > capacity_) {
      storage_.blockPop(size - capacity_);
    }
    batch.seqFirst_();
    batch.to_(device);
    return batch;
  }

  const int prefetch_;
  const int capacity_;

  // make sure that multiple calls of sample does not overlap
  std::unique_ptr<std::thread> samplerThread_;
  // basic concurrent queue for read and write data
  std::queue<RNNTransition> samples_;
  std::mutex mSampler_;
  std::condition_variable cvSampler_;

  ConcurrentQueue storage_;
  std::atomic<int> numAdd_;

  std::mt19937 rng_;
};

}
