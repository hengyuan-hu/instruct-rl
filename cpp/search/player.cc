#include "cpp/search/player.h"

namespace search {

// observe before act
void Player::observeBeforeAct(const GameSimulator& env) {
  assert(futBp_.isNull());
  auto input = observe(env.state(), index);
  addHid(input, bpHid_);

  if (!llmPrior_.empty()) {
    auto lastMove = getLastNonDealMoveFromState(env.state(), index);

    float piklLambda = 0;
    auto foundMove = llmPrior_.end();

    if (env.state().CurPlayer() == index) {
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
  }

  futBp_ = bpModel_->call("act", input);
}

int Player::decideAction(const GameSimulator& env) {
  // first get results from the futures, to update hid
  auto bpReply = futBp_.get();
  moveHid(bpReply, bpHid_);
  int action = bpReply.at("a").item<int64_t>();

  if (env.state().CurPlayer() != index) {
    assert(action == env.game().MaxMoves());
  }
  return action;
}
}  // namespace search
