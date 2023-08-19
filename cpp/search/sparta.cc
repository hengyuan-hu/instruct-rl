#include <chrono>
#include <future>

#include "cpp/search/game_sim.h"
#include "cpp/search/sparta.h"
#include "cpp/utils.h"

namespace search {

std::vector<std::vector<hle::HanabiCardValue>> filterSample(
    const torch::Tensor& samples,
    const std::vector<int>& privCardCount,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand) {
  auto sampleAcc = samples.accessor<int64_t, 2>();
  int numSample = sampleAcc.size(0);
  int handSize = hand.Cards().size();

  std::vector<std::vector<hle::HanabiCardValue>> ret;

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
      cards.push_back(card);
    }
    if ((int)cards.size() == handSize && hand.CanSetCards(cards)) {
      ret.push_back(cards);
    }
  }
  return ret;
}

float searchMove(
    const hle::HanabiState& state,
    hle::HanabiMove move,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int>& seeds,
    int myIdx,
    const std::vector<Player>& players) {
  std::vector<std::vector<Player>> allPlayers(hands.size(), players);
  std::vector<GameSimulator> games;
  games.reserve(hands.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    std::vector<SimHand> simHands{
        SimHand(myIdx, hands[i]),
    };
    games.emplace_back(state, simHands, seeds[i]);
  }

  size_t terminated = 0;
  std::vector<int> notTerminated(games.size());
  std::iota(notTerminated.begin(), notTerminated.end(), 0);

  bool searchMoveApplied = false;
  while (!notTerminated.empty()) {
    std::vector<int> newNotTerminated;
    for (auto i : notTerminated) {
      assert(!games[i].state().IsTerminal());
      for (auto& actor : allPlayers[i]) {
        actor.observeBeforeAct(games[i]);
      }
    }

    for (auto i : notTerminated) {
      auto& game = games[i];
      int action = -1;
      for (auto& actor : allPlayers[i]) {
        int a = actor.decideAction(game);
        if (actor.index == game.state().CurPlayer()) {
          action = a;
        }
      }

      if (!searchMoveApplied) {
        game.step(move);
      } else {
        game.step(game.getMove(action));
      }

      if (!game.terminal()) {
        newNotTerminated.push_back(i);
      } else {
        ++terminated;
      }
    }

    notTerminated = newNotTerminated;
    if (!searchMoveApplied) {
      searchMoveApplied = true;
    }
  }
  assert(terminated == games.size());

  std::vector<float> scores(games.size());
  float mean = 0;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(games[i].terminal());
    scores[i] = games[i].score();
    mean += scores[i];
  }
  mean = mean / scores.size();
  return mean;
}

std::vector<float> parallelSearchMoves(
    const hle::HanabiState& state,
    const std::vector<hle::HanabiMove>& moves,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int>& seeds,
    int myIdx,
    const std::vector<Player>& players) {
  std::vector<float> scores;
  std::vector<std::future<float>> futs;
  for (auto& move : moves) {
    futs.push_back(std::async(
        std::launch::async, searchMove, state, move, hands, seeds, myIdx, players));
  }

  for (auto& fut : futs) {
    float score = fut.get();
    scores.push_back(score);
  }
  return scores;
}
}  // namespace search
