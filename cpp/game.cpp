#include "game.h"
#include "Zobrist.h"
#include <algorithm>
#include <iostream>

void Game::setState(const std::vector<int>& board, const std::vector<int>& controlled, int round, int player) {
    board_ = board;
    controlled_ = controlled;
    round_ = round;
    current_player_ = player;

    // State has changed completely, so we must fully recalculate the hash
    hash_ = 0;
    hash_ ^= ZobristTable::getInstance().getRoundHash(round_);
    for (int r = 0; r < size_; ++r) {
        for (int c = 0; c < size_; ++c) {
            int idx = getIndex(r, c);
            if (board_[idx] == 1) hash_ ^= ZobristTable::getInstance().getHash(r, c, 0);
            if (board_[idx] == -1) hash_ ^= ZobristTable::getInstance().getHash(r, c, 1);
            if (controlled_[idx] == 1) hash_ ^= ZobristTable::getInstance().getHash(r, c, 2);
            if (controlled_[idx] == -1) hash_ ^= ZobristTable::getInstance().getHash(r, c, 3);
        }
    }
}

Game::Game(int size, int max_rounds)
    : size_(size),
    current_player_(1),
    game_over_(false),
    winner_(0),
    round_(0),
    max_rounds_(max_rounds),
    hash_(0) {
    // Zobrist initialization is correctly handled in GameInterface constructor.
    // We just get the initial hash for round 0.
    hash_ = ZobristTable::getInstance().getRoundHash(0);
    board_.assign(size_ * size_, 0);
    controlled_.assign(size_ * size_, 0);
}

inline int Game::getIndex(int r, int c) const { return r * size_ + c; }
inline int Game::getIndex(const Position& pos) const { return pos.r * size_ + pos.c; }
uint64_t Game::getHash() const { return hash_; }
bool Game::isGameOver() const { return game_over_; }
int Game::getWinner() const { return winner_; }
int Game::getCurrentPlayer() const { return current_player_; }
const std::vector<int>& Game::getBoard() const { return board_; }
const std::vector<int>& Game::getControlled() const { return controlled_; }
int Game::getRound() const { return round_; }

std::vector<Position> Game::getValidMoves() const {
    std::vector<Position> valid_moves;
    if (game_over_) return valid_moves;
    for (int r = 0; r < size_; ++r) {
        for (int c = 0; c < size_; ++c) {
            int idx = getIndex(r, c);
            if (board_[idx] == 0 && (controlled_[idx] == 0 || controlled_[idx] == current_player_)) {
                valid_moves.push_back({ r, c });
            }
        }
    }
    return valid_moves;
}

bool Game::makeMove(const Position& pos) {
    if (game_over_ || pos.r < 0 || pos.r >= size_ || pos.c < 0 || pos.c >= size_) return false;
    int idx = getIndex(pos);
    if (board_[idx] != 0 || (controlled_[idx] != 0 && controlled_[idx] != current_player_)) return false;

    board_[idx] = current_player_;
    int piece_type = (current_player_ == 1) ? 0 : 1;
    hash_ ^= ZobristTable::getInstance().getHash(pos.r, pos.c, piece_type);

    checkThreeConnection(pos);

    hash_ ^= ZobristTable::getInstance().getRoundHash(round_);
    round_++;

    if (round_ < max_rounds_) {
        hash_ ^= ZobristTable::getInstance().getRoundHash(round_);
    }

    if (round_ >= max_rounds_) {
        game_over_ = true;
        checkWinner();
    }

    current_player_ = -current_player_;
    return true;
}

void Game::checkThreeConnection(const Position& pos) {
    int player = board_[getIndex(pos)];
    int opponent = -player;
    const std::vector<std::pair<std::string, std::vector<std::pair<int, int>>>> directions = {
        {"horizontal", {{0, 1}, {0, -1}}}, {"vertical", {{1, 0}, {-1, 0}}},
        {"diagonal", {{1, 1}, {-1, -1}}}, {"anti_diagonal", {{1, -1}, {-1, 1}}}
    };
    std::vector<std::pair<std::string, std::vector<Position>>> lines_to_control;
    std::set<Position> stones_to_remove;
    for (const auto& dir_info : directions) {
        std::vector<Position> line = findConnectedLine(pos, dir_info.second, player);
        if (line.size() >= 3) {
            lines_to_control.push_back({ dir_info.first, line });
            for (const auto& p : line) stones_to_remove.insert(p);
        }
    }
    if (lines_to_control.empty()) return;

    std::set<Position> controlled_positions;
    for (auto& line_info : lines_to_control) {
        std::set<Position> controlled = getControlledPositions(line_info.second, line_info.first, player, opponent);
        controlled_positions.insert(controlled.begin(), controlled.end());
    }
    int control_piece_type = (player == 1) ? 2 : 3;
    for (const auto& p : controlled_positions) {
        int c_idx = getIndex(p);
        if (controlled_[c_idx] != player) {
            if (controlled_[c_idx] == opponent) {
                int old_control_type = (player == 1) ? 3 : 2;
                hash_ ^= ZobristTable::getInstance().getHash(p.r, p.c, old_control_type);
            }
            controlled_[c_idx] = player;
            hash_ ^= ZobristTable::getInstance().getHash(p.r, p.c, control_piece_type);
        }
    }
    int stone_piece_type = (player == 1) ? 0 : 1;
    for (const auto& p : stones_to_remove) {
        if (board_[getIndex(p)] != 0) {
            board_[getIndex(p)] = 0;
            hash_ ^= ZobristTable::getInstance().getHash(p.r, p.c, stone_piece_type);
        }
    }
}

std::vector<Position> Game::findConnectedLine(const Position& pos, const std::vector<std::pair<int, int>>& dir_pair, int player) const {
    std::vector<Position> line;
    line.push_back(pos);
    for (const auto& dir : dir_pair) {
        Position current_pos = pos;
        while (true) {
            current_pos.r += dir.first;
            current_pos.c += dir.second;
            if (current_pos.r < 0 || current_pos.r >= size_ || current_pos.c < 0 || current_pos.c >= size_ || board_[getIndex(current_pos)] != player) break;
            line.push_back(current_pos);
        }
    }
    return line;
}

std::set<Position> Game::getControlledPositions(std::vector<Position>& line, const std::string& dir_name, int player, int opponent) const {
    std::set<Position> controlled;
    std::sort(line.begin(), line.end());
    if (dir_name == "horizontal") {
        int row = line[0].r;
        int min_col = line[0].c, max_col = line.back().c;
        int left_end = min_col, right_end = max_col;
        for (int c = min_col - 1; c >= 0; --c) { if (board_[getIndex(row, c)] == opponent) break; left_end = c; }
        for (int c = max_col + 1; c < size_; ++c) { if (board_[getIndex(row, c)] == opponent) break; right_end = c; }
        for (int c = left_end; c <= right_end; ++c) controlled.insert({ row, c });
    }
    else if (dir_name == "vertical") {
        int col = line[0].c;
        int min_row = line[0].r, max_row = line.back().r;
        int top_end = min_row, bottom_end = max_row;
        for (int r = min_row - 1; r >= 0; --r) { if (board_[getIndex(r, col)] == opponent) break; top_end = r; }
        for (int r = max_row + 1; r < size_; ++r) { if (board_[getIndex(r, col)] == opponent) break; bottom_end = r; }
        for (int r = top_end; r <= bottom_end; ++r) controlled.insert({ r, col });
    }
    else if (dir_name == "diagonal") {
        int constant = line[0].r - line[0].c;
        int min_r = line[0].r, max_r = line.back().r;
        int start_r = min_r, end_r = max_r;
        for (int r = min_r - 1; r >= 0; --r) { int c = r - constant; if (c < 0 || c >= size_ || board_[getIndex(r, c)] == opponent) break; start_r = r; }
        for (int r = max_r + 1; r < size_; ++r) { int c = r - constant; if (c < 0 || c >= size_ || board_[getIndex(r, c)] == opponent) break; end_r = r; }
        for (int r = start_r; r <= end_r; ++r) { int c = r - constant; if (c >= 0 && c < size_) controlled.insert({ r, c }); }
    }
    else if (dir_name == "anti_diagonal") {
        int constant = line[0].r + line[0].c;
        int min_r = line[0].r, max_r = line.back().r;
        int start_r = min_r, end_r = max_r;
        for (int r = min_r - 1; r >= 0; --r) { int c = constant - r; if (c < 0 || c >= size_ || board_[getIndex(r, c)] == opponent) break; start_r = r; }
        for (int r = max_r + 1; r < size_; ++r) { int c = constant - r; if (c < 0 || c >= size_ || board_[getIndex(r, c)] == opponent) break; end_r = r; }
        for (int r = start_r; r <= end_r; ++r) { int c = constant - r; if (c >= 0 && c < size_) controlled.insert({ r, c }); }
    }
    return controlled;
}

void Game::checkWinner() {
    int black_control = 0, white_control = 0;
    for (int i = 0; i < size_ * size_; ++i) {
        if (controlled_[i] == 1) black_control++;
        else if (controlled_[i] == -1) white_control++;
    }
    if (black_control > white_control) winner_ = 1;
    else if (white_control > black_control) winner_ = -1;
    else winner_ = 0;
}

int Game::getScore() const {
    int black_control = 0, white_control = 0;
    for (int i = 0; i < size_ * size_; ++i) {
        if (controlled_[i] == 1) black_control++;
        else if (controlled_[i] == -1) white_control++;
    }
    return black_control - white_control;
}
