#include "cpp/search/game_sim.h"

using namespace search;

void GameSimulator::reset(
    const hle::HanabiState& refState, const std::vector<SimHand>& simHands) {
  state_ = refState;
  state_.SetGame(&game_);
  terminal_ = false;
  reward_ = 0;

  for (const auto& simHand : simHands) {
    const auto& realCards = state_.Hands()[simHand.index].Cards();
    auto& deck = state_.Deck();
    deck.PutCardsBack(realCards);
  }
  for (const auto& simHand : simHands) {
    auto& deck = state_.Deck();
    deck.DealCards(simHand.cards);

    auto& hand = state_.Hands()[simHand.index];
    if (!hand.CanSetCards(simHand.cards)) {
      std::cout << "cannot set hand:" << std::endl;
      std::cout << "real hand: " << std::endl;
      std::cout << hand.ToString() << std::endl;
      std::cout << "sim hand: ";
      for (auto& c : simHand.cards) {
        std::cout << c.ToString() << ", ";
      }
      std::cout << std::endl;
    }
    hand.SetCards(simHand.cards);
  }
}
