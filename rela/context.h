// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <atomic>
#include <cassert>
#include <memory>
#include <thread>
#include <vector>

#include "rela/thread_loop.h"

namespace rela {

class Context {
 public:
  Context()
      : started_(false)
      , numTerminatedThread_(0) {
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ~Context();

  void reset();

  int pushThreadLoop(std::shared_ptr<ThreadLoop> env);

  void start();

  void pause();

  void resume();

  void join();

  bool terminated();

 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<ThreadLoop>> loops_;
  std::vector<std::thread> threads_;
};
}  // namespace rela
