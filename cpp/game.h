#ifndef GAME_H
#define GAME_H

#include <vector>
#include <utility>
#include <string>
#include <set>
#include <cstdint> // For uint64_t

struct Position {
    int r, c;
    bool operator<(const Position& other) const {
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};

class Game {
public:
    Game(int size = 9, int max_rounds = 50);

    std::vector<Position> getValidMoves() const;
    bool makeMove(const Position& pos);
    bool isGameOver() const;
    int getWinner() const;
    int getScore() const;
    int getCurrentPlayer() const;
    const std::vector<int>& getBoard() const;
    const std::vector<int>& getControlled() const;
    int getRound() const;

    // 获取当前哈希值
    uint64_t getHash() const;

    void setState(const std::vector<int>& board, const std::vector<int>& controlled, int round, int player);

private:
    int size_;
    std::vector<int> board_;
    std::vector<int> controlled_;
    int current_player_;
    bool game_over_;
    int winner_;
    int round_;
    int max_rounds_;
    friend class GameInterface;

    // 当前棋盘状态的 Zobrist 哈希值
    uint64_t hash_;

    int getIndex(int r, int c) const;
    int getIndex(const Position& pos) const;
    void checkThreeConnection(const Position& pos);
    std::vector<Position> findConnectedLine(const Position& pos, const std::vector<std::pair<int, int>>& dir_pair, int player) const;
    std::set<Position> getControlledPositions(std::vector<Position>& line, const std::string& dir_name, int player, int opponent) const;
    void checkWinner();
};

#endif // GAME_H
