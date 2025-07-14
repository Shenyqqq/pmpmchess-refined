#include "GameInterface.h"
#include "Zobrist.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <iostream>

namespace GI_Debug {
    void log(const std::string& msg) { std::cerr << "[GI_CPP] " << msg << std::endl; }
    void log_hash(const std::string& tag, uint64_t hash) { std::cerr << "[GI_CPP] " << tag << ": " << hash << std::endl; }
    void log_float(const std::string& tag, float val) { std::cerr << "[GI_CPP] " << tag << ": " << val << std::endl; }
    void log_int(const std::string& tag, int val) { std::cerr << "[GI_CPP] " << tag << ": " << val << std::endl; }
}

bool GameInterface::logging_enabled_ = false;

void rotate_plane_90_clockwise(const float* in, float* out, int n) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            out[c * n + (n - 1 - r)] = in[r * n + c];
        }
    }
}

void flip_plane_horizontal(const float* in, float* out, int n) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            out[r * n + (n - 1 - c)] = in[r * n + c];
        }
    }
}


GameInterface::GameInterface(int n, int max_rounds)
    : n_(n), max_rounds_(max_rounds), board_channels_(13) {
    // =================================================================
    // ===== BUGFIX: Zobrist table is now initialized here, and ONLY here. =====
    // =================================================================
    ZobristTable::getInstance().initialize(n, max_rounds);
    GI_Debug::log("Zobrist Initialize");
}

void GameInterface::setLogging(bool enabled) {
    logging_enabled_ = enabled;
    if (logging_enabled_) {
        std::cerr << "[GameInterface] C++ logging for Arena has been ENABLED." << std::endl;
    }
    else {
        std::cerr << "[GameInterface] C++ logging for Arena has been DISABLED." << std::endl;
    }
}

std::pair<int, int> GameInterface::getBoardSize() const {
    return { n_, n_ };
}

int GameInterface::getActionSize() const {
    return n_ * n_;
}

std::pair<BoardState, uint64_t> GameInterface::getInitialBoard() const {
    Game logic_game(n_, max_rounds_);
    auto result = std::make_pair(packState(logic_game), logic_game.getHash());
    return result;
}

std::tuple<BoardState, int, uint64_t> GameInterface::getNextState(const BoardState& board, int player, int action) const {
    Game logic_game = unpackState(board, player);
    logic_game.makeMove({ action / n_, action % n_ });
    auto result = std::make_tuple(packState(logic_game), -player, logic_game.getHash());
    return result;
}

std::pair<BoardState, uint64_t> GameInterface::getCanonicalForm(const BoardState& board, uint64_t hash, int player) const {
    if (player == 1) {
        return { board, hash };
    }

    BoardState canonical_board = board;
    int size = n_ * n_;
    const std::vector<std::pair<int, int>> channels_to_swap = { {0, 1}, {2, 3}, {5, 6}, {7, 8}, {11, 12} };
    for (const auto& p : channels_to_swap) {
        int ch1_start = p.first * size;
        int ch2_start = p.second * size;
        for (int i = 0; i < size; ++i) {
            std::swap(canonical_board[ch1_start + i], canonical_board[ch2_start + i]);
        }
    }

    Game canonical_game = unpackState(canonical_board, 1);
    uint64_t canonical_hash = canonical_game.getHash();

    return { canonical_board, canonical_hash };
}

BoardState GameInterface::packState(const Game& logic_game) const {
    BoardState state(n_ * n_ * board_channels_, 0.0f);
    const auto& board = logic_game.getBoard();
    const auto& controlled = logic_game.getControlled();
    int current_player = logic_game.getCurrentPlayer();
    auto get_idx = [&](int r, int c, int channel) { return channel * (n_ * n_) + r * n_ + c; };

    int current_round = logic_game.getRound();
    float round_val = static_cast<float>(current_round) / max_rounds_;
    int black_controlled_count = 0;
    int white_controlled_count = 0;
    for (int r = 0; r < n_; ++r) {
        for (int c = 0; c < n_; ++c) {
            int flat_idx = r * n_ + c;
            if (board[flat_idx] == 1) state[get_idx(r, c, 0)] = 1.0f;
            if (board[flat_idx] == -1) state[get_idx(r, c, 1)] = 1.0f;
            if (controlled[flat_idx] == 1) { state[get_idx(r, c, 2)] = 1.0f; black_controlled_count++; }
            if (controlled[flat_idx] == -1) { state[get_idx(r, c, 3)] = 1.0f; white_controlled_count++; }
            if (board[flat_idx] == 0) state[get_idx(r, c, 9)] = 1.0f;
            if (controlled[flat_idx] == 0) state[get_idx(r, c, 10)] = 1.0f;
        }
    }
    Game temp_game = logic_game;
    std::vector<float> black_valid_moves(n_ * n_, 0.0f);
    temp_game.current_player_ = 1;
    for (const auto& move : temp_game.getValidMoves()) { black_valid_moves[move.r * n_ + move.c] = 1.0f; }
    std::vector<float> white_valid_moves(n_ * n_, 0.0f);
    temp_game.current_player_ = -1;
    for (const auto& move : temp_game.getValidMoves()) { white_valid_moves[move.r * n_ + move.c] = 1.0f; }

    float player_to_move_val = (current_player == 1) ? 1.0f : 0.0f;
    float black_control_norm = static_cast<float>(black_controlled_count) / (n_ * n_);
    float white_control_norm = static_cast<float>(white_controlled_count) / (n_ * n_);
    for (int r = 0; r < n_; ++r) {
        for (int c = 0; c < n_; ++c) {
            int flat_idx = r * n_ + c;
            state[get_idx(r, c, 4)] = round_val;
            state[get_idx(r, c, 5)] = player_to_move_val;
            state[get_idx(r, c, 6)] = 1.0f - player_to_move_val;
            state[get_idx(r, c, 7)] = black_control_norm;
            state[get_idx(r, c, 8)] = white_control_norm;
            state[get_idx(r, c, 11)] = black_valid_moves[flat_idx];
            state[get_idx(r, c, 12)] = white_valid_moves[flat_idx];
        }
    }
    return state;
}
Game GameInterface::unpackState(const BoardState& state, int player) const {
    Game logic_game(n_, max_rounds_); // 1. 创建一个临时的、干净的游戏对象
    auto get_idx = [&](int r, int c, int channel) { return channel * (n_ * n_) + r * n_ + c; };

    // =================================================================
    // ===== BUGFIX START: Use a public method to set the state =====
    // =================================================================

    // 2. 从 state vector 中提取棋盘和控制区信息
    std::vector<int> board_data(n_ * n_, 0);
    std::vector<int> controlled_data(n_ * n_, 0);
    for (int r = 0; r < n_; ++r) {
        for (int c = 0; c < n_; ++c) {
            int flat_idx = r * n_ + c;
            if (state[get_idx(r, c, 0)] == 1.0f) board_data[flat_idx] = 1;
            if (state[get_idx(r, c, 1)] == 1.0f) board_data[flat_idx] = -1;
            if (state[get_idx(r, c, 2)] == 1.0f) controlled_data[flat_idx] = 1;
            if (state[get_idx(r, c, 3)] == 1.0f) controlled_data[flat_idx] = -1;
        }
    }

    // 3. 提取回合数
    int round = static_cast<int>(state[get_idx(0, 0, 4)] * max_rounds_);

    // 4. 调用新的公共接口来安全地设置游戏状态，该接口会负责正确地重建哈希
    logic_game.setState(board_data, controlled_data, round, player);

    return logic_game;
    // =================================================================
    // ===== BUGFIX END ================================================
    // =================================================================
}

std::vector<int> GameInterface::getValidMoves(const BoardState& board, int player) const {
    std::vector<int> valid_moves(getActionSize(), 0);
    int channel = (player == 1) ? 11 : 12;
    for (int i = 0; i < getActionSize(); ++i) { if (board[channel * n_ * n_ + i] == 1.0f) { valid_moves[i] = 1; } }
    return valid_moves;
}

float GameInterface::getGameEnded(const BoardState& board, int player) const {
    if (static_cast<int>(board[4 * n_ * n_] * max_rounds_) < max_rounds_) return 0.0f;
    float p1_score = 0, p2_score = 0;
    for (int i = 0; i < n_ * n_; ++i) { p1_score += board[2 * n_ * n_ + i]; p2_score += board[3 * n_ * n_ + i]; }

    if (p1_score > p2_score) return (player == 1) ? 1.0f : -1.0f;
    if (p2_score > p1_score) return (player == -1) ? 1.0f : -1.0f;
    return 1e-4f;
}

float GameInterface::getScore(const BoardState& board, int player) const {
    float p1_score = 0, p2_score = 0;
    for (int i = 0; i < n_ * n_; ++i) { p1_score += board[2 * n_ * n_ + i]; p2_score += board[3 * n_ * n_ + i]; }
    return (player == 1) ? (p1_score - p2_score) : (p2_score - p1_score);
}

std::vector<std::pair<BoardState, std::vector<float>>> GameInterface::getSymmetries(const BoardState& board, const std::vector<float>& pi) const {
    std::vector<std::pair<BoardState, std::vector<float>>> symmetries;
    symmetries.reserve(8);

    BoardState current_board = board;
    std::vector<float> current_pi = pi;

    BoardState temp_board_storage(board.size());
    std::vector<float> temp_pi_storage(pi.size());

    for (int i = 0; i < 4; ++i) {
        symmetries.emplace_back(current_board, current_pi);

        BoardState flipped_board(board.size());
        std::vector<float> flipped_pi(pi.size());

        flip_plane_horizontal(current_pi.data(), flipped_pi.data(), n_);

        for (int ch = 0; ch < board_channels_; ++ch) {
            const float* current_channel_ptr = current_board.data() + ch * n_ * n_;
            float* flipped_channel_ptr = flipped_board.data() + ch * n_ * n_;
            flip_plane_horizontal(current_channel_ptr, flipped_channel_ptr, n_);
        }
        symmetries.emplace_back(flipped_board, flipped_pi);

        if (i < 3) {
            rotate_plane_90_clockwise(current_pi.data(), temp_pi_storage.data(), n_);
            current_pi = temp_pi_storage;

            for (int ch = 0; ch < board_channels_; ++ch) {
                const float* current_channel_ptr = current_board.data() + ch * n_ * n_;
                float* temp_channel_ptr = temp_board_storage.data() + ch * n_ * n_;
                rotate_plane_90_clockwise(current_channel_ptr, temp_channel_ptr, n_);
            }
            current_board = temp_board_storage;
        }
    }
    return symmetries;
}

