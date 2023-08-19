#include "cpp/r2d2_actor.h"
#include "cpp/utils.h"
#include <iomanip>

std::vector<hle::HanabiCardValue> sampleCards(
    const std::vector<float>& v0,
    const std::vector<int>& privCardCount,
    const std::vector<int>& invColorPermute,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand,
    std::mt19937& rng) {
  auto handSize = hand.Cards().size();
  auto cardBelief = extractPerCardBelief(v0, game, handSize);
  auto cardRemain = privCardCount;
  std::vector<hle::HanabiCardValue> cards;

  for (size_t j = 0; j < handSize; ++j) {
    auto& cb = cardBelief[j];
    if (j > 0) {
      // re-mask card belief
      float sum = 0;
      for (size_t k = 0; k < cardRemain.size(); ++k) {
        cb[k] *= int(cardRemain[k] > 0);
        sum += cb[k];
      }

      if (sum <= 1e-6) {
        std::cerr << "Error in sample card, sum = 0" << std::endl;
        assert(false);
      }
    }
    std::discrete_distribution<int> dist(cb.begin(), cb.end());
    int idx = dist(rng);
    --cardRemain[idx];
    assert(cardRemain[idx] >= 0);
    if (invColorPermute.size()) {
      auto fakeColor = indexToCard(idx, game.NumRanks());
      auto realColor =
          hle::HanabiCardValue(invColorPermute[fakeColor.Color()], fakeColor.Rank());
      cards.push_back(realColor);
    } else {
      cards.push_back(indexToCard(idx, game.NumRanks()));
    }
  }

  assert(hand.CanSetCards(cards));
  return cards;
}

std::tuple<std::vector<hle::HanabiCardValue>, bool> filterSample(
    const torch::Tensor& samples,
    const std::vector<int>& privCardCount,
    const std::vector<int>& invColorPermute,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand) {
  auto sampleAcc = samples.accessor<int64_t, 2>();
  int numSample = sampleAcc.size(0);
  int handSize = hand.Cards().size();

  for (int i = 0; i < numSample; ++i) {
    auto cardRemain = privCardCount;
    std::vector<hle::HanabiCardValue> cards;
    for (int j = 0; j < handSize; ++j) {
      // sampling & v0 belief is done in the color shuffled space
      int idx = sampleAcc[i][j];
      auto card = indexToCard(idx, game.NumRanks());
      // this sample violate card count
      if (cardRemain[idx] == 0) {
        break;
      }
      --cardRemain[idx];

      if (invColorPermute.size()) {
        auto realCard = hle::HanabiCardValue(invColorPermute[card.Color()], card.Rank());
        cards.push_back(realCard);
      } else {
        cards.push_back(card);
      }
    }
    if ((int)cards.size() == handSize && hand.CanSetCards(cards)) {
      return {cards, true};
    }
  }
  return {hand.CardValues(), false};
}

std::tuple<bool, bool> analyzeCardBelief(const std::vector<float>& b) {
  assert(b.size() == 25);
  std::set<int> colors;
  std::set<int> ranks;
  for (int c = 0; c < 5; ++c) {
    for (int r = 0; r < 5; ++r) {
      if (b[c * 5 + r] > 0) {
        colors.insert(c);
        ranks.insert(r);
      }
    }
  }
  return {colors.size() == 1, ranks.size() == 1};
}

rela::RNNTransition stackForVDN(const std::vector<rela::RNNTransition>& taus) {
  assert(taus.size() > 1);

  rela::RNNTransition tauJoint;
  tauJoint.reward = taus[0].reward;
  // tauJoint.terminal = taus[0].terminal;
  tauJoint.bootstrap = taus[0].bootstrap;
  tauJoint.seqLen = taus[0].seqLen;

  std::vector<rela::TensorDict> allObs;
  std::vector<rela::TensorDict> allAction;
  std::vector<rela::TensorDict> allH0;
  for (size_t i = 0; i < taus.size(); ++i) {
    allObs.push_back(taus[i].obs);
    allAction.push_back(taus[i].action);
    allH0.push_back(taus[i].h0);

    if (i > 0) {
      assert(tauJoint.reward.equal(taus[i].reward));
      // assert(tauJoint.terminal.equal(taus[i].terminal));
      assert(tauJoint.bootstrap.equal(taus[i].bootstrap));
      assert(tauJoint.seqLen.equal(taus[i].seqLen));
    }
  }

  tauJoint.obs = rela::tensor_dict::stack(allObs, 1);
  tauJoint.action = rela::tensor_dict::stack(allAction, 1);
  tauJoint.h0 = rela::tensor_dict::cat(allH0, 1);

  return tauJoint;
}

/////////////////////////// R2D2Actor ///////////////////////////////

void R2D2Actor::reset(const HanabiEnv& env) {
  assert(futures_.empty());  // should have no outstanding futures
  assert(stepDone());        // should not be in the middle of a step

  // some asserts to keep us sane
  if (r2d2Buffer_ == nullptr) {
    assert(!offBelief_);
  }
  if (offBelief_) {
    assert(!vdn_);
  }

  hidden_ = initHidden_;
  // for (auto& kv : hidden_) {
  //   assert(kv.second.sum().item<float>() != 0);
  // }
  if (beliefRunner_ != nullptr) {
    assert(false);  // disabled
    beliefHidden_ = getH0(1, beliefRunner_);
  }

  if (r2d2Buffer_ != nullptr) {
    r2d2Buffer_->init(hidden_);
  }

  if (epsList_.size()) {
    assert(playerEps_.size() == 1);
    playerEps_[0] = epsList_[rng_() % epsList_.size()];
  }

  if (tempList_.size() > 0) {
    assert(playerTemp_.size() == 0);
    playerTemp_[0] = tempList_[rng_() % tempList_.size()];
  }

  if (piklLambdas_.size() > 0) {
    piklLambda_ = piklLambdas_[rng_() % piklLambdas_.size()];
  }

  // other-play
  if (shuffleColor_) {
    const auto& game = env.getHleGame();
    colorPermute_.clear();
    invColorPermute_.clear();
    for (int i = 0; i < game.NumColors(); ++i) {
      colorPermute_.push_back(i);
      invColorPermute_.push_back(i);
    }
    std::shuffle(colorPermute_.begin(), colorPermute_.end(), rng_);
    std::sort(invColorPermute_.begin(), invColorPermute_.end(), [&](int i, int j) {
      return colorPermute_[i] < colorPermute_[j];
    });
    // std::cout << "color permute: ";
    for (int i = 0; i < (int)colorPermute_.size(); ++i) {
      // std::cout << "(" << i << "->" << colorPermute_[i] << ")";
      assert(invColorPermute_[colorPermute_[i]] == i);
    }
    // std::cout << std::endl;
  }
}

bool R2D2Actor::ready() const {
  for (auto& kv : futures_) {
    if (!kv.second.isReady()) {
      return false;
    }
  }
  return true;
}

bool R2D2Actor::stepDone() const {
  return stage_ == Stage::ObserveBeforeAct;
}

// Returns optional move and whether we're done acting
// std::tuple<std::optional<hle::HanabiMove>, bool>
std::unique_ptr<hle::HanabiMove> R2D2Actor::next(const HanabiEnv& env) {
  if (stage_ == Stage::ObserveBeforeAct) {
    assert(futures_.size() == 0);
    // std::cout << "observe before act" << std::endl;
    observeBeforeAct(env);
    stage_ = Stage::DecideMove;
    return nullptr;
  }

  if (stage_ == Stage::DecideMove) {
    // std::cout << "deciding move" << std::endl;
    auto move = decideMove(env);
    if (offBelief_) {
      stage_ = Stage::FictAct;
    } else {
      if (replayBuffer_ != nullptr) {
        stage_ = Stage::ObserveAfterAct;
      } else {
        stage_ = Stage::ObserveBeforeAct;
      }
    }
    return move;
  }

  if (stage_ == Stage::FictAct) {
    // std::cout << "fict act" << std::endl;
    fictAct(env);

    stage_ = Stage::ObserveAfterAct;
    return nullptr;
  }

  if (stage_ == Stage::ObserveAfterAct) {
    // std::cout << "observe after act" << std::endl;
    observeAfterAct(env);

    if (env.terminated()) {
      stage_ = Stage::StoreTrajectory;
    } else {
      stage_ = Stage::ObserveBeforeAct;
    }
    return nullptr;
  }

  if (stage_ == Stage::StoreTrajectory) {
    // std::cout << "store trajectory" << std::endl;
    storeTrajectory(env);
    stage_ = Stage::ObserveBeforeAct;
    return nullptr;
  }

  assert(false);
  return nullptr;
}

void R2D2Actor::observeBeforeAct(const HanabiEnv& env) {
  torch::NoGradGuard ng;
  prevHidden_ = hidden_;

  const auto& state = env.getHleState();
  auto input = observe(
      state,
      playerIdx_,
      shuffleColor_,
      colorPermute_,
      invColorPermute_,
      hideAction_,
      aux_,
      sad_);

  // add features such as eps and temperature
  if (epsList_.size()) {
    input["eps"] = torch::tensor(playerEps_);
  }
  if (playerTemp_.size() > 0) {
    input["temperature"] = torch::tensor(playerTemp_);
  }

  if (llmPrior_.size()) {
    auto obs = hle::HanabiObservation(state, playerIdx_, true);
    auto lastMove = getLastNonDealMove(obs.LastMoves());

    float piklLambda = 0;
    auto foundMove = llmPrior_.end();
    if (env.getCurrentPlayer() == playerIdx_) {
      piklLambda = piklLambda_;
      if (lastMove == nullptr) {
        // std::cout << "Prev Move: None" << std::endl;
        foundMove = llmPrior_.find("[null]");
      } else {
        // std::cout << "Prev Move: " << lastMove->ToLangKey() << std::endl;
        foundMove = llmPrior_.find(lastMove->ToLangKey());
      }
    } else {
      piklLambda = 0;
      foundMove = llmPrior_.find("[null]");
    }

    assert(foundMove != llmPrior_.end());
    input["pikl_lambda"] = torch::tensor(piklLambda, torch::kFloat32);
    // this contains just the scaled logits
    input["llm_prior"] = foundMove->second * piklBeta_;
    // currentLLMPrior_ = foundMove->second * piklBeta_;
  }

  if (replayBuffer_ != nullptr) {
    // collect aoh push before we add hidden
    r2d2Buffer_->pushObs(input);
  }

  addHid(input, hidden_);
  // no-blocking async call to neural network
  futures_["act"] = runner_->call("act", input);

  if (replayBuffer_ == nullptr) {
    // eval mode, collect some stats
    const auto& game = env.getHleGame();
    auto obs = hle::HanabiObservation(state, state.CurPlayer(), true);
    auto encoder = hle::CanonicalObservationEncoder(&game);
    auto [privV0, cardCount] = encoder.EncodeV0Belief(
      obs, std::vector<int>(), false, std::vector<int>(), false);
    perCardPrivV0_ = extractPerCardBelief(
      privV0, env.getHleGame(), obs.Hands()[0].Cards().size());
  }

  if (!offBelief_) {
    return;
  }

  // forward belief model
  assert(!shuffleColor_ && ! hideAction_);
  auto [beliefInput, privCardCount, v0] = spartaObserve(
      state, playerIdx_);
  privCardCount_ = privCardCount;

  if (beliefRunner_ == nullptr) {
    sampledCards_ = sampleCards(
        v0,
        privCardCount_,
        invColorPermute_,
        env.getHleGame(),
        state.Hands()[playerIdx_],
        rng_);
  } else {
    addHid(beliefInput, beliefHidden_);
    futures_["belief"] = beliefRunner_->call("sample", beliefInput);
  }

  fictState_ = std::make_unique<hle::HanabiState>(state);
}

std::unique_ptr<hle::HanabiMove> R2D2Actor::decideMove(const HanabiEnv& env) {
  torch::NoGradGuard ng;

  // get act results, update hid, and recard action if needed
  auto reply = getResultAndErase("act", futures_);
  int action = reply.at("a").item<int64_t>();
  moveHid(reply, hidden_);

  // if (llmPrior_.size() > 0 && env.getCurrentPlayer() == playerIdx_) {
  //   auxReward_ = currentLLMPrior_[action].item<float>();
  //   // std::cout << "last action" << env.getMove(env.getLastAction()).ToString()
  //   //           << ", current action: " << env.getMove(action).ToString() << std::endl;
  //   // std::cout << "adding additional reward: " << auxReward_ << std::endl;
  // } else {
  //   auxReward_ = 0;
  // }

  if (env.getCurrentPlayer() == playerIdx_ && reply.count("bp_logits") > 0) {
    auto bp_logits = reply.at("bp_logits");
    auto adv = reply.at("adv");
    auto pikl_lambda = reply.at("pikl_lambda");
    auto legal_adv = reply.at("legal_adv");
    auto legal_move = reply.at("legal_move");

    // std::cout << "legal_adv: " << legal_adv.sizes() << std::endl;
    std::cout << "@decideMove, step: " << env.numStep() << std::endl;
    std::cout << "@decideMove last move: " << env.getMove(env.getLastAction()).ToString() << std::endl;
    for (int action = 0; action < 20; ++action) {
      if (legal_move[action].item<int>() == 0) {
        continue;
      }
      if (pikl_lambda.item<float>() == 0) {
        continue;
      }
      std::cout << std::fixed;
      std::cout << std::setprecision(4)
                << "@decideMove action: " << env.getMove(action).ToString()
                << ", adv: " << adv[action].item<float>()
                << ", bp_logits: " << bp_logits[action].item<float>()
                << ", final_adv: " << legal_adv[action].item<float>() << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;
  } else if (env.getCurrentPlayer() == playerIdx_ && reply.count("adv") > 0) {
    std::cout << "@decideMove, step: " << env.numStep() << std::endl;
    auto adv = reply.at("adv");
    auto legal_move = reply.at("legal_move");

    std::cout << "@decideMove last move: " << env.getMove(env.getLastAction()).ToString() << std::endl;
    for (int action = 0; action < 20; ++action) {
      if (legal_move[action].item<int>() == 0) {
        continue;
      }
      std::cout << std::fixed;
      std::cout << std::setprecision(4)
                << "@decideMove action: " << env.getMove(action).ToString()
                << ", adv: " << adv[action].item<float>() << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;
  }

  if (replayBuffer_ != nullptr) {
    r2d2Buffer_->pushAction(reply);
  }

  // get the real action
  auto curPlayer = env.getCurrentPlayer();
  std::unique_ptr<hle::HanabiMove> move;
  if (curPlayer != playerIdx_) {
    assert(action == env.noOpUid());
  } else {
    auto& state = env.getHleState();
    move = std::make_unique<hle::HanabiMove>(state.ParentGame()->GetMove(action));
    if (shuffleColor_ && move->MoveType() == hle::HanabiMove::Type::kRevealColor) {
      int realColor = invColorPermute_[move->Color()];
      move->SetColor(realColor);
    }

    if (replayBuffer_ == nullptr) {
      // collect stats
      if (move->MoveType() == hle::HanabiMove::kPlay) {
        auto cardBelief = perCardPrivV0_[move->CardIndex()];
        auto [colorKnown, rankKnown] = analyzeCardBelief(cardBelief);

        if (colorKnown && rankKnown) {
          ++bothKnown_;
        } else if (colorKnown) {
          ++colorKnown_;
        } else if (rankKnown) {
          ++rankKnown_;
        } else {
          ++noneKnown_;
        }
      }
    }
  }

  if (offBelief_) {
    const auto& hand = fictState_->Hands()[playerIdx_];
    bool success = true;

    if (beliefRunner_ != nullptr) {
      auto beliefReply = getResultAndErase("belief", futures_);
      moveHid(beliefReply, beliefHidden_);
      auto sample = beliefReply.at("sample");
      std::tie(sampledCards_, success) = filterSample(
          sample,
          privCardCount_,
          invColorPermute_,
          env.getHleGame(),  // *fictGame_,
          hand);
    }

    if (success) {
      auto& deck = fictState_->Deck();
      deck.PutCardsBack(hand.Cards());
      deck.DealCards(sampledCards_);
      fictState_->Hands()[playerIdx_].SetCards(sampledCards_);
      ++successFict_;
    }
    validFict_ = success;
    ++totalFict_;

    if (curPlayer != playerIdx_) {
      auto partner = partners_[curPlayer];
      assert(partner != nullptr);
      // it is not my turn, I need to re-evaluate my partner on
      // the fictitious transition
      auto partnerInput = observe(
          *fictState_,
          partner->playerIdx_,
          partner->shuffleColor_,
          partner->colorPermute_,
          partner->invColorPermute_,
          partner->hideAction_,
          partner->aux_,
          partner->sad_);
      // add features such as eps and temperature
      partnerInput["eps"] = torch::tensor(partner->playerEps_);
      if (partner->playerTemp_.size() > 0) {
        partnerInput["temperature"] = torch::tensor(partner->playerTemp_);
      }
      addHid(partnerInput, partner->prevHidden_);
      futures_["fict_act"] = partner->runner_->call("act", partnerInput);
    }
  }
  return move;
}

void R2D2Actor::fictAct(const HanabiEnv& env) {
  if (!offBelief_) {
    return;
  }

  assert(futures_.size() == 0 || futures_.size() == 1);
  auto fictMove = env.lastMove();
  if (futures_.size() == 1) {
    auto fictReply = getResultAndErase("fict_act", futures_);
    auto action = fictReply["a"].item<int64_t>();
    fictMove = env.getMove(action);
  }

  auto [fictR, fictTerm] = applyMove(*fictState_, fictMove, false);

  // submit network call to compute target value
  auto fictInput = observe(
      *fictState_,
      playerIdx_,
      shuffleColor_,
      colorPermute_,
      invColorPermute_,
      hideAction_,
      aux_,
      sad_);
  addHid(fictInput, hidden_);

  // the hidden is new, so we are good
  fictInput["reward"] = torch::tensor(fictR);
  fictInput["terminal"] = torch::tensor(float(fictTerm));
  if (playerTemp_.size() > 0) {
    fictInput["temperature"] = torch::tensor(playerTemp_);
  }

  futures_["target"] = runner_->call("compute_target", fictInput);
}

void R2D2Actor::observeAfterAct(const HanabiEnv& env) {
  if (replayBuffer_ == nullptr) {
    return;
  }

  float reward = env.stepReward();
  // reward += auxReward_;
  bool terminated = env.terminated();
  r2d2Buffer_->pushReward(reward);
  r2d2Buffer_->pushTerminal(float(terminated));

  if (offBelief_) {
    assert(false);
    // assert(futures_.size() == 1);
    // auto target = getResultAndErase("target", futures_).at("target");
    // r2d2Buffer_->obsBack().emplace("target", target);
    // r2d2Buffer_->obsBack().emplace("valid_fict", torch::tensor(float(validFict_)));
  }
  assert(futures_.size() == 0);

  if (!env.terminated()) {
    return;
  }
  // std::cout << "---------terminal state hit-----------" << std::endl;
  // if (playerIdx_ == numPlayer_ - 1) {
  //   assert(false);
  // }

  // submit priority call at the end of the episode
  if (vdn_) {
    // if vdn, we need to collect AOH from other players,
    // ONLY the LAST player will collect and aggregate data from all players
    if (playerIdx_ != numPlayer_ - 1) {
      return;
    }

    std::vector<rela::RNNTransition> taus;
    assert(numPlayer_ == (int)partners_.size());
    for (int i = 0; i < numPlayer_ - 1; ++i) {
      taus.push_back(partners_[i]->r2d2Buffer_->popTransition());
    }
    taus.push_back(r2d2Buffer_->popTransition());
    lastEpisode_ = stackForVDN(taus);
  } else {
    lastEpisode_ = r2d2Buffer_->popTransition();
  }

  // auto input = lastEpisode_.toDict();
  // futures_["priority"] = runner_->call("compute_priority", input);
}

void R2D2Actor::storeTrajectory(const HanabiEnv& env) {
  if (!env.terminated()) {
    return;
  }

  if (vdn_ && playerIdx_ != numPlayer_ - 1) {
    return;
  }

  assert(futures_.size() == 0);
  // auto priority = getResultAndErase("priority", futures_).at("priority").item<float>();
  replayBuffer_->add(lastEpisode_);
}
