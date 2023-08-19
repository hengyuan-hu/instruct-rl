#pragma once

#include "rela/batch_runner.h"
#include "rela/replay.h"
#include "rela/r2d2.h"

#include "cpp/hanabi_env.h"
#include "cpp/utils.h"

class R2D2Actor {
 public:
  R2D2Actor(
      std::shared_ptr<rela::BatchRunner> runner,
      int seed,
      int numPlayer, // total number os players
      int playerIdx, // player idx for this player
      bool vdn,
      bool sad,
      bool shuffleColor,
      bool hideAction,
      AuxType aux,   // trinary aux task or full aux
      std::shared_ptr<rela::Replay> replayBuffer,
      // if replay buffer is None, then all params below are not used
      int multiStep,
      int seqLen,
      float gamma)
      : runner_(std::move(runner))
      , rng_(seed)
      , numPlayer_(numPlayer)
      , playerIdx_(playerIdx)
      , vdn_(vdn)
      , sad_(sad)
      , shuffleColor_(shuffleColor)
      , hideAction_(hideAction)
      , aux_(aux)
      , playerEps_(1)
      , playerTemp_(1)
      , replayBuffer_(std::move(replayBuffer))
      , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(multiStep, seqLen, gamma)) {
    initHidden_ = getH0(1, runner_);
  }

  // simpler constructor for eval mode
  R2D2Actor(
      std::shared_ptr<rela::BatchRunner> runner,
      int numPlayer,
      int playerIdx,
      bool vdn,
      bool sad,
      bool hideAction)
      : runner_(std::move(runner))
      , rng_(1)  // not used in eval mode
      , numPlayer_(numPlayer)
      , playerIdx_(playerIdx)
      , vdn_(vdn)
      , sad_(sad)
      , shuffleColor_(false)
      , hideAction_(hideAction)
      , aux_(AuxType::Null)
      , playerEps_(1)
      , playerTemp_(1)
      , replayBuffer_(nullptr)
      , r2d2Buffer_(nullptr) {
    initHidden_ = getH0(1, runner_);
  }

  void setPartners(std::vector<std::shared_ptr<R2D2Actor>> partners) {
    partners_ = std::move(partners);
    assert((int)partners_.size() == numPlayer_);
    assert(partners_[playerIdx_] == nullptr);
  }

  void setExploreEps(std::vector<float> eps) {
    epsList_ = std::move(eps);
  }

  void setBoltzmannT(std::vector<float> t) {
    tempList_ = std::move(t);
  }

  void setLLMPrior(
      const std::unordered_map<std::string, std::vector<float>>& llmPrior,
      std::vector<float> piklLambdas,
      float piklBeta) {
    assert(llmPrior_.size() == 0); // has not been previously set
    for (const auto& kv : llmPrior) {
      llmPrior_[kv.first] = torch::tensor(kv.second, torch::kFloat32);
    }
    piklLambdas_ = std::move(piklLambdas);
    piklBeta_ = piklBeta;
    // assert(piklBeta == 1);
  }

  void updateLLMLambda(std::vector<float> piklLambdas) {
    piklLambdas_ = std::move(piklLambdas);
  }

  void reset(const HanabiEnv& env);

  bool ready() const;

  bool stepDone() const;

  std::unique_ptr<hle::HanabiMove> next(const HanabiEnv& env);

  void setBeliefRunner(std::shared_ptr<rela::BatchRunner>& beliefModel) {
    beliefRunner_ = beliefModel;  // this can be a nullptr, i.e. analytical belief
    offBelief_ = true;
    // OBL does not need Other-Play, and does not support Other-Play
    assert(!shuffleColor_);
  }

  float getSuccessFictRate() {
    float rate = -1;
    if (totalFict_) {
      rate = (float)successFict_ / totalFict_;
    }
    successFict_ = 0;
    totalFict_ = 0;
    return rate;
  }

  std::tuple<int, int, int, int> getPlayedCardInfo() const {
    return {noneKnown_, colorKnown_, rankKnown_, bothKnown_};
  }

  enum class Stage {
    ObserveBeforeAct,
    DecideMove,
    FictAct,
    ObserveAfterAct,
    StoreTrajectory
  };

 private:
  rela::TensorDict getH0(int numPlayer, std::shared_ptr<rela::BatchRunner>& runner) {
    std::vector<torch::jit::IValue> input{numPlayer};
    auto model = runner->jitModel();
    auto output = model.get_method("get_h0")(input);
    auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
    return h0;
  }

  void observeBeforeAct(const HanabiEnv& env);

  std::unique_ptr<hle::HanabiMove> decideMove(const HanabiEnv& env);

  void fictAct(const HanabiEnv& env);

  void observeAfterAct(const HanabiEnv& env);

  void storeTrajectory(const HanabiEnv& env);

  std::shared_ptr<rela::BatchRunner> runner_;
  std::mt19937 rng_;
  const int numPlayer_;
  const int playerIdx_;
  const bool vdn_;
  const bool sad_;
  const bool shuffleColor_;
  const bool hideAction_;
  const AuxType aux_;

  // optional, e.g. ppo does not use it
  std::vector<float> epsList_;
  std::vector<float> tempList_;

  std::vector<float> playerEps_;  // vector for easy conversion to tensor, size==1
  std::vector<float> playerTemp_;
  std::vector<int> colorPermute_;
  std::vector<int> invColorPermute_;

  std::shared_ptr<rela::Replay> replayBuffer_;
  std::unique_ptr<rela::R2D2Buffer> r2d2Buffer_;

  rela::TensorDict initHidden_;
  rela::TensorDict prevHidden_;
  rela::TensorDict hidden_;

  rela::RNNTransition lastEpisode_;

  bool offBelief_ = false;
  std::shared_ptr<rela::BatchRunner> beliefRunner_;
  rela::TensorDict beliefHidden_;

  std::vector<int> privCardCount_;
  std::vector<hle::HanabiCardValue> sampledCards_;

  int totalFict_ = 0;
  int successFict_ = 0;
  bool validFict_ = false;
  std::unique_ptr<hle::HanabiState> fictState_ = nullptr;
  std::vector<std::shared_ptr<R2D2Actor>> partners_;

  // to control stages
  Stage stage_ = Stage::ObserveBeforeAct;
  std::unordered_map<std::string, rela::FutureReply> futures_;

  // information on cards played
  // only computed during eval mode (replayBuffer==nullptr)
  std::vector<std::vector<float>> perCardPrivV0_;
  int noneKnown_ = 0;
  int colorKnown_ = 0;
  int rankKnown_ = 0;
  int bothKnown_ = 0;

  // llm stuff
  std::unordered_map<std::string, torch::Tensor> llmPrior_;
  std::vector<float> piklLambdas_;
  float piklLambda_ = 0;
  float piklBeta_ = 1;

  // torch::Tensor currentLLMPrior_;
  // float auxReward_ = 0;
};

