#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <vector>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <iostream> 

namespace Zobrist_Debug {
    inline void log(const std::string& msg) { std::cerr << "[ZOBRIST_CPP] " << msg << std::endl; }
}

constexpr int PIECE_TYPES = 4;

class ZobristTable {
public:
    static ZobristTable& getInstance() {
        static ZobristTable instance;
        return instance;
    }

    uint64_t getHash(int r, int c, int piece_type) const {
        if (r < 0 || r >= size_ || c < 0 || c >= size_ || piece_type < 0 || piece_type >= PIECE_TYPES) {
            throw std::out_of_range("Zobrist table access out of range for piece.");
        }
        return table_[r][c][piece_type];
    }

    uint64_t getRoundHash(int round) const {
        if (round < 0 || round > max_rounds_) { 
            throw std::out_of_range("Zobrist table access out of range for round.");
        }
        return round_hashes_[round];
    }

    void initialize(int board_size, int max_rounds) {
        if (initialized_) {
            // Zobrist_Debug::log("Initialize called but already initialized. Skipping.");
            return;
        }
        Zobrist_Debug::log("Initialize START. board_size=" + std::to_string(board_size) + ", max_rounds=" + std::to_string(max_rounds));
        size_ = board_size;
        max_rounds_ = max_rounds;
        std::mt19937_64 rng(1337);

        table_.resize(size_, std::vector<std::vector<uint64_t>>(size_, std::vector<uint64_t>(PIECE_TYPES)));
        for (int r = 0; r < size_; ++r) {
            for (int c = 0; c < size_; ++c) {
                for (int p = 0; p < PIECE_TYPES; ++p) {
                    table_[r][c][p] = rng();
                }
            }
        }

        round_hashes_.resize(max_rounds_ + 1);
        for (int i = 0; i <= max_rounds_; ++i) { 
            round_hashes_[i] = rng();
        }

        initialized_ = true;
        Zobrist_Debug::log("Initialize END. Table is ready.");
    }

private:
    ZobristTable() : size_(0), max_rounds_(0), initialized_(false) {}
    ZobristTable(const ZobristTable&) = delete;
    ZobristTable& operator=(const ZobristTable&) = delete;

    int size_;
    int max_rounds_;
    bool initialized_;
    std::vector<std::vector<std::vector<uint64_t>>> table_;
    std::vector<uint64_t> round_hashes_;
};

#endif // ZOBRIST_H
