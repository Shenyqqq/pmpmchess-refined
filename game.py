import numpy as np
from typing import Tuple, List, Optional, Set
from visualizer import GameVisualizer
class Game:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)# 棋盘状态 0=无，1=黑占，-1=白占
        self.controlled = np.zeros((size, size), dtype=int)# 占领区域
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.round = 0
        self.max_rounds = 50
        self.control_black = 0
        self.control_white = 0
        self.n = size

    def _update_control_counts(self):
        """更新控制区域计数"""
        self.control_black = np.sum(self.controlled == 1)
        self.control_white = np.sum(self.controlled == -1)
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """返回可落子位置"""
        return [(i, j) for i in range(self.size)
                for j in range(self.size)
                if self.board[i, j] == 0 and
                (self.controlled[i, j] == 0 or
                 self.controlled[i, j] == self.current_player)] #未落子、未占领或自己占领

    def make_move(self, pos: Tuple[int, int]) -> bool:
        """落子"""
        if self.game_over or pos not in self.get_valid_moves():
            return False

        i, j = pos
        self.board[i, j] = self.current_player
        self._check_three_connection(pos)
        self._update_control_counts()

        self.round += 1
        if self.round >= self.max_rounds:
            self.game_over = True
            self.check_winner()

        self.current_player = -self.current_player # 切换玩家
        return True

    def _check_three_connection(self, pos: Tuple[int, int]):
        """检查所有方向的连三并统一处理占领"""
        directions = [
            ('horizontal', (0, 1), (0, -1)),  # 水平
            ('vertical', (1, 0), (-1, 0)),  # 垂直
            ('diagonal', (1, 1), (-1, -1)),  # 主对角线
            ('anti_diagonal', (1, -1), (-1, 1))  # 副对角线
        ]

        player = self.board[pos]
        opponent = -player
        lines_to_control = []
        stones_to_remove = set()

        # 1. 收集所有连三的线
        for dir_name, dir1, dir2 in directions:
            line = self._find_connected_line(pos, [dir1, dir2], player)
            if len(line) >= 3:
                lines_to_control.append((dir_name, line))
                stones_to_remove.update(line)

        # 2. 统一处理所有占领线
        controlled_positions = set()
        for dir_name, line in lines_to_control:
            # 处理每条线的占领范围
            controlled = self._get_controlled_positions(line, dir_name, player, opponent)
            controlled_positions.update(controlled)

        # 3. 执行占领和清除棋子
        for i, j in controlled_positions:
            if self.controlled[i, j] != player:
                self.controlled[i, j] = player
        for i, j in stones_to_remove:
            self.board[i, j] = 0

    def _find_connected_line(self, pos, dir_pair, player) -> List[Tuple[int, int]]:
        """沿两个方向搜索连续同色棋子"""
        line = [pos]
        for di, dj in dir_pair:
            i, j = pos
            while True:
                i, j = i + di, j + dj
                if not (0 <= i < self.size and 0 <= j < self.size):
                    break
                if self.board[i, j] != player:
                    break
                line.append((i, j))
        return line

    def _get_controlled_positions(self, line, dir_name, player, opponent) -> Set[Tuple[int, int]]:
        """获取一条线可以占领的所有位置"""
        line = sorted(line)
        controlled = set()

        if dir_name == 'horizontal':
            row = line[0][0]
            min_col = min(p[1] for p in line)
            max_col = max(p[1] for p in line)

            left_end = min_col
            for j in range(min_col - 1, -1, -1):
                if self.board[row, j] == opponent:
                    break
                left_end = j

            right_end = max_col
            for j in range(max_col + 1, self.size):
                if self.board[row, j] == opponent:
                    break
                right_end = j

            controlled.update((row, j) for j in range(left_end, right_end + 1))

        elif dir_name == 'vertical':
            col = line[0][1]
            min_row = min(p[0] for p in line)
            max_row = max(p[0] for p in line)

            top_end = min_row
            for i in range(min_row - 1, -1, -1):
                if self.board[i, col] == opponent:
                    break
                top_end = i

            bottom_end = max_row
            for i in range(max_row + 1, self.size):
                if self.board[i, col] == opponent:
                    break
                bottom_end = i

            controlled.update((i, col) for i in range(top_end, bottom_end + 1))

        elif dir_name == 'diagonal':  # 主对角线
            const = line[0][0] - line[0][1]
            min_i = min(p[0] for p in line)
            max_i = max(p[0] for p in line)

            # 西北
            start_i = min_i
            for i in range(min_i - 1, -1, -1):
                j = i - const
                if not (0 <= j < self.size) or self.board[i, j] == opponent:
                    break
                start_i = i

            # 东南
            end_i = max_i
            for i in range(max_i + 1, self.size):
                j = i - const
                if not (0 <= j < self.size) or self.board[i, j] == opponent:
                    break
                end_i = i

            controlled.update((i, i - const) for i in range(start_i, end_i + 1) if 0 <= (i - const) < self.size)

        elif dir_name == 'anti_diagonal':  # 副对角线
            const = line[0][0] + line[0][1]
            min_i = min(p[0] for p in line)
            max_i = max(p[0] for p in line)

            # 东北
            start_i = min_i
            for i in range(min_i - 1, -1, -1):
                j = const - i
                if not (0 <= j < self.size) or self.board[i, j] == opponent:
                    break
                start_i = i

            # 西南
            end_i = max_i
            for i in range(max_i + 1, self.size):
                j = const - i
                if not (0 <= j < self.size) or self.board[i, j] == opponent:
                    break
                end_i = i

            controlled.update((i, const - i) for i in range(start_i, end_i + 1) if 0 <= (const - i) < self.size)

        return controlled

    def check_winner(self) -> Optional[int]:
        """根据占领区域数量判断胜负"""
        if self.round >= self.max_rounds:
            self.game_over = True
            black_control = np.sum(self.controlled == 1)
            white_control = np.sum(self.controlled == -1)

            if black_control > white_control :
                self.winner = 1
            elif white_control > black_control:
                self.winner = -1
            else:
                self.winner = 0

    def score(self):
        black_control = np.sum(self.controlled == 1)
        white_control = np.sum(self.controlled == -1)
        return black_control - white_control


    def visualize(self):
        visualizer = GameVisualizer(self)
        visualizer.run()

    def display(self,board):
        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.board[board[:, :, 0] == 1] = 1
        self.board[board[:, :, 1] == 1] = -1
        self.controlled = np.zeros((self.n, self.n), dtype=np.int8)
        self.controlled[board[:, :, 2] == 1] = 1
        self.controlled[board[:, :, 3] == 1] = -1
        print("Board")
        print(self.board)
        print("Control")
        print(self.controlled)



if __name__ == "__main__":
    game = Game(size=9)
    game.visualize()
