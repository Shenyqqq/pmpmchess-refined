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
// --- 新增头文件 ---
#include <thread>
#include <future>
#include <mutex>

namespace py = pybind11;

struct MCTSArgs {
    int numMCTSSims = 200;
    float cpuct = 1.0;
    float dirichletAlpha = 0.1;
    float epsilon = 0.25;
    float factor_winloss = 1.0;
    int num_threads = 1; // 新增：用于MCTS的线程数
};

class MCTS {
public:
    MCTS(const GameInterface& game, py::object predictor, const MCTSArgs& args);

    // 公开接口，现在是多线程调度器
    py::list getActionProbs(const py::list& canonicalBoards, const py::list& hashes, const py::list& seeds, float temp);

private:
    const GameInterface& game_;
    py::object predictor_;
    MCTSArgs args_;
    int action_size_;
    std::mt19937 rng_; // 主线程的RNG，每个工作线程将使用自己的RNG

    // --- 新增的Python GIL锁 ---
    // 用于保护对Python对象的调用，确保线程安全
    std::mutex gil_mutex_;

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

    // --- 新增的私有方法，由每个线程执行 ---
    // 这个函数包含了原来 getActionProbs 的单线程MCTS搜索逻辑
    py::list search_batch(const py::list& canonicalBoards_py, const py::list& hashes_py, const py::list& seeds_py, float temp);

    // 静态成员函数保持不变
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
