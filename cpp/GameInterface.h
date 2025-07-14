#ifndef GAME_INTERFACE_H
#define GAME_INTERFACE_H

#include "game.h"
#include <vector>
#include <string>
#include <cstdint>
#include <tuple>

using BoardState = std::vector<float>;

class GameInterface {
public:
    GameInterface(int n = 9, int max_rounds = 50);

    std::pair<BoardState, uint64_t> getInitialBoard() const;
    std::pair<int, int> getBoardSize() const;
    int getActionSize() const;
    std::tuple<BoardState, int, uint64_t> getNextState(const BoardState& board, int player, int action) const;
    std::vector<int> getValidMoves(const BoardState& board, int player) const;
    float getGameEnded(const BoardState& board, int player) const;

    // getCanonicalForm 现在返回一个包含同步后哈希值的 pair
    std::pair<BoardState, uint64_t> getCanonicalForm(const BoardState& board, uint64_t hash, int player) const;

    float getScore(const BoardState& board, int player) const;
    std::vector<std::pair<BoardState, std::vector<float>>> getSymmetries(const BoardState& board, const std::vector<float>& pi) const;
    static void setLogging(bool enabled);

private:
    int n_;
    int max_rounds_;
    int board_channels_;

    BoardState packState(const Game& logic_game) const;
    Game unpackState(const BoardState& board, int player) const;
    static bool logging_enabled_;
};

#endif // GAME_INTERFACE_H
