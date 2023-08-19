// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rela/batch_runner.h"
#include "rela/context.h"
#include "rela/replay.h"
// #include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"
#include "rela/transition.h"

namespace py = pybind11;
using namespace rela;

PYBIND11_MODULE(rela, m) {
  py::class_<RNNTransition, std::shared_ptr<RNNTransition>>(m, "RNNTransition")
      .def_readwrite("obs", &RNNTransition::obs)
      .def_readwrite("h0", &RNNTransition::h0)
      .def_readwrite("action", &RNNTransition::action)
      .def_readwrite("reward", &RNNTransition::reward)
    //   .def_readwrite("terminal", &RNNTransition::terminal)
      .def_readwrite("bootstrap", &RNNTransition::bootstrap)
      .def_readwrite("seq_len", &RNNTransition::seqLen);


  // py::class_<RNNPrioritizedReplay, std::shared_ptr<RNNPrioritizedReplay>>(
  //     m, "RNNPrioritizedReplay")
  //     .def(py::init<
  //          int,    // capacity,
  //          int,    // seed,
  //          float,  // alpha, priority exponent
  //          float,  // beta, importance sampling exponent
  //          int>())
  //     .def("clear", &RNNPrioritizedReplay::clear)
  //     .def("reset_alpha", &RNNPrioritizedReplay::resetAlpha)
  //     .def("terminate", &RNNPrioritizedReplay::terminate)
  //     .def("size", &RNNPrioritizedReplay::size)
  //     .def("num_add", &RNNPrioritizedReplay::numAdd)
  //     .def("sample", &RNNPrioritizedReplay::sample)
  //     .def("update_priority", &RNNPrioritizedReplay::updatePriority)
  //     .def("get", &RNNPrioritizedReplay::get)
  //     .def("get_first_k", &RNNPrioritizedReplay::getFirstK)
  //     .def("get_range", &RNNPrioritizedReplay::getRange);

  py::class_<Replay, std::shared_ptr<Replay>>(
      m, "RNNReplay")
      .def(py::init<
           int,    // capacity,
           int,    // seed,
           int>())
      .def("clear", &Replay::clear)
      .def("terminate", &Replay::terminate)
      .def("size", &Replay::size)
      .def("num_add", &Replay::numAdd)
      .def("sample", &Replay::sample)
      .def("get", &Replay::get)
      .def("get_range", &Replay::getRange);

//   py::class_<TensorDictReplay, std::shared_ptr<TensorDictReplay>>(m, "TensorDictReplay")
//       .def(py::init<
//            int,    // capacity,
//            int,    // seed,
//            float,  // alpha, priority exponent
//            float,  // beta, importance sampling exponent
//            int>())
//       .def("size", &TensorDictReplay::size)
//       .def("num_add", &TensorDictReplay::numAdd)
//       .def("sample", &TensorDictReplay::sample)
//       .def("update_priority", &TensorDictReplay::updatePriority)
//       .def("get", &TensorDictReplay::get);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("reset", &Context::reset)
      .def("push_thread_loop", &Context::pushThreadLoop)//, py::keep_alive<1, 2>())
      .def("start", &Context::start)
      .def("pause", &Context::pause)
      .def("resume", &Context::resume)
      .def("join", &Context::join)
      .def("terminated", &Context::terminated);

  py::class_<BatchRunner, std::shared_ptr<BatchRunner>>(m, "BatchRunner")
      .def(py::init<
           py::object,
           const std::string&,
           int,
           const std::vector<std::string>&>())
      .def(py::init<py::object, const std::string&>())
      .def("add_method", &BatchRunner::addMethod)
      .def("start", &BatchRunner::start)
      .def("stop", &BatchRunner::stop)
      .def("acquire_model_lock", &BatchRunner::acquireModelLock)
      .def("release_model_lock", &BatchRunner::releaseModelLock)
      .def("update_model", &BatchRunner::updateModel)
      .def("set_log_freq", &BatchRunner::setLogFreq)
      .def("log_and_clear_agg_size", &BatchRunner::logAndClearAggSize);
}
