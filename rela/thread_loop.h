// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include <cassert>

namespace rela {

class ThreadLoop {
 public:
  ThreadLoop() = default;

  ThreadLoop(const ThreadLoop&) = delete;
  ThreadLoop& operator=(const ThreadLoop&) = delete;

  virtual ~ThreadLoop() {
  }

  virtual void terminate() {
    terminated_ = true;
    if (paused()) {
      resume();
    }
  }

  virtual void pause() {
    std::lock_guard<std::mutex> lk(mPaused_);
    assert(!paused_);  // you cannot pause twice
    if (numWait_ != 0) {
      std::cout << numWait_ << " threads are still waiting." << std::endl;
      assert(false);
    }
    paused_ = true;
  }

  virtual void resume() {
    std::lock_guard<std::mutex> lk(mPaused_);
    paused_ = false;
    cvPaused_.notify_all();
  }

  virtual void waitUntilResume() {
    std::unique_lock<std::mutex> lk(mPaused_);
    numWait_ += 1;
    cvPaused_.wait(lk, [this] { return !paused_; });
    numWait_ -= 1;
  }

  virtual bool terminated() {
    return terminated_;
  }

  virtual bool paused() {
    return paused_;
  }

  virtual void mainLoop() = 0;

 private:
  std::atomic_bool terminated_{false};

  std::mutex mPaused_;
  bool paused_ = false;
  std::condition_variable cvPaused_;
  int numWait_ = 0;
};

}  // namespace rela
