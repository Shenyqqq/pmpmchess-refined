#ifndef MCTS_H
#define MCTS_H

#include "GameInterface.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <cstdint>

namespace py = pybind11;

struct MCTSArgs {
    int numMCTSSims = 200;
    float cpuct = 1.0;
    float dirichletAlpha = 0.1;
    float epsilon = 0.25;
    float factor_winloss = 1.0;
};

class MCTS {
public:
    MCTS(const GameInterface& game, py::object predictor, const MCTSArgs& args);
    py::list getActionProbs(const py::list& canonicalBoards, const py::list& hashes, const py::list& seeds, float temp);
private:

private:
    const GameInterface& game_;
    py::object predictor_;
    MCTSArgs args_;
    int action_size_;
    std::mt19937 rng_;

    using PolicyMap = std::unordered_map<uint64_t, std::vector<float>>;
    using QValueMap = std::unordered_map<uint64_t, std::vector<float>>;
    using VisitCountMap = std::unordered_map<uint64_t, std::vector<int>>;
    using StateVisitMap = std::unordered_map<uint64_t, int>;
    using ValidMovesMap = std::unordered_map<uint64_t, std::vector<int>>;

    struct SearchPath {
        std::vector<std::pair<uint64_t, int>> path;
        BoardState leaf_state;
        uint64_t leaf_hash;
    };

    static SearchPath findLeaf(
        const BoardState& board,
        uint64_t hash,
        const GameInterface& game,
        int action_size,
        float cpuct,
        PolicyMap& Ps,
        QValueMap& Qsa,
        VisitCountMap& Nsa,
        StateVisitMap& Ns,
        ValidMovesMap& Vs
    );

    static void addDirichletNoise(
        uint64_t hash,
        const MCTSArgs& args,
        std::mt19937& rng,
        PolicyMap& Ps,
        const ValidMovesMap& Vs
    );
};

#endif // MCTS_H
