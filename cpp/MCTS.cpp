#define _USE_MATH_DEFINES
#include "MCTS.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread> // 确保包含

namespace MCTS_Debug {
    void log(const std::string& msg) {
        std::cout << "[MCTS_DEBUG] " << msg << std::endl;
    }
}

MCTS::MCTS(const GameInterface& game, py::object predictor, const MCTSArgs& args)
    : game_(game), predictor_(predictor), args_(args), action_size_(game.getActionSize()) {
    // 初始化主线程的RNG，但每个工作线程会用自己的seed重新初始化
    rng_ = std::mt19937(std::random_device{}());
}

// ===================================================================================
// 新的多线程调度器 getActionProbs (保持不变)
// ===================================================================================
py::list MCTS::getActionProbs(const py::list& canonicalBoards_py, const py::list& hashes_py, const py::list& seeds_py, float temp) {
    size_t total_games = py::len(canonicalBoards_py);
    int num_threads_to_use = std::min((int)total_games, args_.num_threads);

    if (num_threads_to_use <= 1) {
        return search_batch(canonicalBoards_py, hashes_py, seeds_py, temp);
    }

    std::vector<std::future<py::list>> futures;
    size_t base_chunk_size = total_games / num_threads_to_use;
    size_t remainder = total_games % num_threads_to_use;
    size_t current_pos = 0;

    for (int i = 0; i < num_threads_to_use; ++i) {
        size_t chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
        if (chunk_size == 0) continue;

        py::list board_chunk, hash_chunk, seed_chunk;
        for (size_t j = 0; j < chunk_size; ++j) {
            board_chunk.append(canonicalBoards_py[current_pos + j]);
            hash_chunk.append(hashes_py[current_pos + j]);
            seed_chunk.append(seeds_py[current_pos + j]);
        }
        current_pos += chunk_size;

        futures.push_back(std::async(std::launch::async,
            &MCTS::search_batch, this,
            board_chunk, hash_chunk, seed_chunk, temp
        ));
    }

    py::list final_policies;
    for (auto& fut : futures) {
        py::list thread_results = fut.get();
        for (auto item : thread_results) {
            final_policies.append(item);
        }
    }

    return final_policies;
}


// ===================================================================================
// search_batch 函数 (已使用高效的批处理进行优化)
// ===================================================================================
py::list MCTS::search_batch(const py::list& canonicalBoards_py, const py::list& hashes_py, const py::list& seeds_py, float temp) {
    PolicyMap Ps; QValueMap Qsa; VisitCountMap Nsa; StateVisitMap Ns; ValidMovesMap Vs;
    std::mt19937 thread_rng;

    // --- 优化的初始节点评估 ---
    py::list boards_for_initial_predict;
    std::vector<uint64_t> hashes_for_initial_predict;

    // 1. 收集所有需要初始评估的棋盘
    for (size_t i = 0; i < py::len(canonicalBoards_py); ++i) {
        uint64_t hash = hashes_py[i].cast<uint64_t>();
        if (Ps.find(hash) == Ps.end()) {
            boards_for_initial_predict.append(canonicalBoards_py[i]);
            hashes_for_initial_predict.push_back(hash);
        }
    }

    // 2. 如果有需要评估的棋盘，进行一次性的批量预测
    if (py::len(boards_for_initial_predict) > 0) {
        py::list predictions;
        { // --- GIL SCOPE START ---
            std::lock_guard<std::mutex> lock(gil_mutex_);
            py::gil_scoped_acquire acquire;
            predictions = predictor_(boards_for_initial_predict);
            py::gil_scoped_release release;
        } // --- GIL SCOPE END ---

        py::array_t<float> policies_py = predictions[0].cast<py::array_t<float>>();
        auto policies = policies_py.unchecked<2>();

        // 3. 解包预测结果并填充MCTS树
        for (size_t i = 0; i < hashes_for_initial_predict.size(); ++i) {
            uint64_t hash = hashes_for_initial_predict[i];

            py::array_t<float> board_py = boards_for_initial_predict[i].cast<py::array_t<float>>();
            BoardState board(board_py.data(), board_py.data() + board_py.size());

            std::vector<float> policy(action_size_);
            for (int k = 0; k < action_size_; ++k) {
                policy[k] = policies(i, k);
            }

            Ps[hash] = policy;
            Vs[hash] = game_.getValidMoves(board, 1);
            float sum_ps_s = 0.0f;
            for (int a = 0; a < action_size_; ++a) { if (Vs[hash][a]) sum_ps_s += Ps[hash][a]; else Ps[hash][a] = 0; }
            if (sum_ps_s > 0) { for (int a = 0; a < action_size_; ++a) Ps[hash][a] /= sum_ps_s; }
            Ns[hash] = 0;
            Qsa[hash] = std::vector<float>(action_size_, 0.0f);
            Nsa[hash] = std::vector<int>(action_size_, 0);
        }
    }

    // 为所有根节点添加狄利克雷噪声
    for (size_t i = 0; i < py::len(hashes_py); ++i) {
        uint64_t hash = hashes_py[i].cast<uint64_t>();
        uint32_t seed = seeds_py[i].cast<uint32_t>();
        thread_rng.seed(seed);
        if (Ps.count(hash)) { // 确保节点已存在
            addDirichletNoise(hash, args_, thread_rng, Ps, Vs);
        }
    }

    // --- MCTS模拟循环 (这部分已经是批量处理，保持不变) ---
    for (int i = 0; i < args_.numMCTSSims; ++i) {
        std::vector<SearchPath> search_paths;
        for (size_t j = 0; j < py::len(canonicalBoards_py); ++j) {
            py::array_t<float> board_py = canonicalBoards_py[j].cast<py::array_t<float>>();
            BoardState board(board_py.data(), board_py.data() + board_py.size());
            uint64_t hash = hashes_py[j].cast<uint64_t>();
            search_paths.push_back(findLeaf(board, hash, game_, action_size_, args_.cpuct, Ps, Qsa, Nsa, Ns, Vs));
        }

        py::list leaves_for_predict;
        for (const auto& sp : search_paths) { leaves_for_predict.append(py::array_t<float>(sp.leaf_state.size(), sp.leaf_state.data())); }

        py::list predictions;
        if (py::len(leaves_for_predict) > 0) {
            { // --- GIL SCOPE START ---
                std::lock_guard<std::mutex> lock(gil_mutex_);
                py::gil_scoped_acquire acquire;
                predictions = predictor_(leaves_for_predict);
                py::gil_scoped_release release;
            } // --- GIL SCOPE END ---
        }
        else {
            continue;
        }

        py::array_t<float> policies_py = predictions[0].cast<py::array_t<float>>();
        py::array_t<float> win_values_py = predictions[1].cast<py::array_t<float>>();
        py::array_t<float> score_values_py = predictions[2].cast<py::array_t<float>>();
        auto policies = policies_py.unchecked<2>();
        auto win_values = win_values_py.unchecked<2>();
        auto score_values = score_values_py.unchecked<2>();

        for (size_t j = 0; j < search_paths.size(); ++j) {
            const auto& sp = search_paths[j];
            uint64_t s_leaf_hash = sp.leaf_hash;
            if (Ps.find(s_leaf_hash) == Ps.end()) {
                std::vector<float> policy(action_size_);
                for (int k = 0; k < action_size_; ++k) policy[k] = policies(j, k);
                Ps[s_leaf_hash] = policy;
                Vs[s_leaf_hash] = game_.getValidMoves(sp.leaf_state, 1);
                float sum_ps_s = 0.0f;
                for (int a = 0; a < action_size_; ++a) { if (Vs[s_leaf_hash][a]) sum_ps_s += Ps[s_leaf_hash][a]; else Ps[s_leaf_hash][a] = 0; }
                if (sum_ps_s > 0) { for (int a = 0; a < action_size_; ++a) Ps[s_leaf_hash][a] /= sum_ps_s; }
                Ns[s_leaf_hash] = 0;
                Qsa[s_leaf_hash] = std::vector<float>(action_size_, 0.0f);
                Nsa[s_leaf_hash] = std::vector<int>(action_size_, 0);
            }
            float game_ended_val = game_.getGameEnded(sp.leaf_state, 1);
            float nn_win_val = win_values(j, 0);
            float nn_score_val = score_values(j, 0);
            float total_utility = (game_ended_val != 0.0f) ? game_ended_val : (args_.factor_winloss * nn_win_val + (2.0f / M_PI) * std::atan(nn_score_val / 2.0f));

            for (auto it = sp.path.rbegin(); it != sp.path.rend(); ++it) {
                total_utility = -total_utility;
                if (Qsa.count(it->first)) {
                    Qsa.at(it->first)[it->second] = (static_cast<float>(Nsa.at(it->first)[it->second]) * Qsa.at(it->first)[it->second] + total_utility) / (Nsa.at(it->first)[it->second] + 1);
                    Nsa.at(it->first)[it->second] += 1;
                    Ns.at(it->first) += 1;
                }
            }
        }
    }

    py::list final_policies;
    for (size_t i = 0; i < py::len(hashes_py); ++i) {
        uint64_t hash = hashes_py[i].cast<uint64_t>();
        std::vector<float> counts(action_size_, 0.0f);
        if (Nsa.count(hash)) {
            const auto& nsa_s = Nsa.at(hash);
            for (int a = 0; a < action_size_; ++a) { counts[a] = static_cast<float>(nsa_s[a]); }
        }
        if (temp == 0.0f) {
            int bestA = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
            std::vector<float> probs(action_size_, 0.0f);
            probs[bestA] = 1.0f;
            final_policies.append(py::array_t<float>(probs.size(), probs.data()));
        }
        else {
            for (auto& count : counts) { count = std::pow(count, 1.0f / temp); }
            float sum_counts = std::accumulate(counts.begin(), counts.end(), 0.0f);
            if (sum_counts > 0) { for (auto& count : counts) { count /= sum_counts; } }
            final_policies.append(py::array_t<float>(counts.size(), counts.data()));
        }
    }
    return final_policies;
}


// findLeaf 和 addDirichletNoise 保持不变
MCTS::SearchPath MCTS::findLeaf(
    const BoardState& board_param, uint64_t hash_param, const GameInterface& game, int action_size, float cpuct,
    PolicyMap& Ps, QValueMap& Qsa, VisitCountMap& Nsa, StateVisitMap& Ns, ValidMovesMap& Vs
) {
    BoardState current_board = board_param;
    uint64_t current_hash = hash_param;
    std::vector<std::pair<uint64_t, int>> path;

    for (int depth = 0; depth < 200; ++depth) {
        if (Ps.find(current_hash) == Ps.end()) {
            return { path, current_board, current_hash };
        }

        auto& valids = Vs.at(current_hash);
        float cur_best = -std::numeric_limits<float>::infinity();
        int best_act = -1;

        for (int a = 0; a < action_size; ++a) {
            if (valids[a]) {
                float qsa = Qsa.at(current_hash)[a];
                int nsa = Nsa.at(current_hash)[a];
                float u = qsa + cpuct * Ps.at(current_hash)[a] * std::sqrt(static_cast<float>(Ns.at(current_hash))) / (1 + nsa);
                if (u > cur_best) { cur_best = u; best_act = a; }
            }
        }

        if (best_act == -1) {
            return { path, current_board, current_hash };
        }

        int action = best_act;
        path.push_back({ current_hash, action });

        auto next_state_info = game.getNextState(current_board, 1, action);

        BoardState next_board_raw = std::get<0>(next_state_info);
        int next_player = std::get<1>(next_state_info);
        uint64_t next_hash_raw = std::get<2>(next_state_info);

        auto canonical_form = game.getCanonicalForm(next_board_raw, next_hash_raw, next_player);
        current_board = canonical_form.first;
        current_hash = canonical_form.second;
    }

    return { path, current_board, current_hash };
}

void MCTS::addDirichletNoise(
    uint64_t hash, const MCTSArgs& args, std::mt19937& rng,
    PolicyMap& Ps, const ValidMovesMap& Vs
) {
    if (args.dirichletAlpha <= 0) return;
    const auto& valids = Vs.at(hash);
    int num_valid_moves = std::accumulate(valids.begin(), valids.end(), 0);
    if (num_valid_moves <= 1) return;

    std::vector<float> dirichlet_samples(num_valid_moves);
    float sum = 0.0f;
    std::gamma_distribution<float> gamma(args.dirichletAlpha, 1.0f);
    for (int i = 0; i < num_valid_moves; ++i) {
        dirichlet_samples[i] = gamma(rng);
        sum += dirichlet_samples[i];
    }
    if (sum < 1e-8) return;
    for (int i = 0; i < num_valid_moves; ++i) {
        dirichlet_samples[i] /= sum;
    }

    auto& policy = Ps.at(hash);
    int sample_idx = 0;
    for (size_t a = 0; a < policy.size(); ++a) {
        if (valids[a]) {
            policy[a] = (1 - args.epsilon) * policy[a] + args.epsilon * dirichlet_samples[sample_idx++];
        }
    }
}
