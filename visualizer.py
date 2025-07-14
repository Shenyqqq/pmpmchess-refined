# visualizer.py
# 使用 Pygame 进行游戏的可视化和交互

import pygame
import numpy as np
import sys

# 尝试导入编译好的 C++ 模块
try:
    import katago_cpp_core
except ImportError:
    print("错误：无法导入 'katago_cpp_core' 模块。")
    print("请先运行 'python setup.py build_ext --inplace' 命令来编译 C++ 代码。")
    sys.exit(1)

# --- 颜色定义 ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (50, 50, 50)
# 棋子颜色
PLAYER_1_COLOR = (220, 50, 50)  # 红色 (黑棋)
PLAYER_2_COLOR = (50, 100, 220)  # 蓝色 (白棋)
# 占领区颜色
PLAYER_1_CONTROL_COLOR = (255, 224, 224)  # 淡红色
PLAYER_2_CONTROL_COLOR = (224, 224, 255)  # 浅蓝色

# --- 棋盘参数 ---
BOARD_SIZE = 9
MAX_ROUNDS = 50
WINDOW_SIZE = 600
MARGIN = 50
CELL_SIZE = (WINDOW_SIZE - 2 * MARGIN) // BOARD_SIZE
PIECE_RADIUS = CELL_SIZE // 2 - 4


class GameVisualizer:
    """游戏可视化和交互处理类"""

    def __init__(self, n=BOARD_SIZE, max_rounds=MAX_ROUNDS):
        """初始化游戏和 Pygame 窗口"""
        self.n = n
        self.game = katago_cpp_core.Game(n, max_rounds)

        # 获取初始棋盘状态
        # FIX: 将返回的 list 转换为 numpy array
        initial_board_list, self.hash = self.game.getInitialBoard()
        self.board = np.array(initial_board_list, dtype=np.float32)

        self.current_player = 1  # 黑棋 (Player 1) 先手
        self.game_over = False
        self.winner = 0

        # 初始化 Pygame
        pygame.init()
        pygame.display.set_caption("游戏可视化")
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.font = pygame.font.SysFont("simhei", 30)  # 使用中文字体

    def draw_board(self):
        """绘制棋盘网格"""
        self.screen.fill(WHITE)
        for i in range(self.n + 1):
            # 垂直线
            start_pos_v = (MARGIN + i * CELL_SIZE, MARGIN)
            end_pos_v = (MARGIN + i * CELL_SIZE, WINDOW_SIZE - MARGIN)
            pygame.draw.line(self.screen, GRID_COLOR, start_pos_v, end_pos_v)
            # 水平线
            start_pos_h = (MARGIN, MARGIN + i * CELL_SIZE)
            end_pos_h = (WINDOW_SIZE - MARGIN, MARGIN + i * CELL_SIZE)
            pygame.draw.line(self.screen, GRID_COLOR, start_pos_h, end_pos_h)

    def draw_pieces_and_territory(self):
        """绘制棋子和双方的占领区"""
        # C++ 模块返回的 board 是一个包含多个通道的扁平化数组
        # 我们需要根据通道索引来获取不同类型的数据
        # 通道 0: 黑棋位置
        # 通道 1: 白棋位置
        # 通道 2: 黑棋占领区
        # 通道 3: 白棋占领区

        # 将扁平化数组重塑为 (通道数, N, N) 的形状，方便索引
        board_reshaped = self.board.reshape(13, self.n, self.n)

        for r in range(self.n):
            for c in range(self.n):
                center_pos = (MARGIN + c * CELL_SIZE + CELL_SIZE // 2,
                              MARGIN + r * CELL_SIZE + CELL_SIZE // 2)

                # 绘制占领区 (作为背景)
                rect = pygame.Rect(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if board_reshaped[2, r, c] == 1.0:  # 黑棋占领区
                    pygame.draw.rect(self.screen, PLAYER_1_CONTROL_COLOR, rect)
                elif board_reshaped[3, r, c] == 1.0:  # 白棋占领区
                    pygame.draw.rect(self.screen, PLAYER_2_CONTROL_COLOR, rect)

        # 重新绘制网格线，使其在占领区颜色之上
        self.draw_board_grid_lines()

        for r in range(self.n):
            for c in range(self.n):
                center_pos = (MARGIN + c * CELL_SIZE + CELL_SIZE // 2,
                              MARGIN + r * CELL_SIZE + CELL_SIZE // 2)

                # 绘制棋子
                if board_reshaped[0, r, c] == 1.0:  # 黑棋 (Player 1)
                    pygame.draw.circle(self.screen, PLAYER_1_COLOR, center_pos, PIECE_RADIUS)
                elif board_reshaped[1, r, c] == 1.0:  # 白棋 (Player 2)
                    pygame.draw.circle(self.screen, PLAYER_2_COLOR, center_pos, PIECE_RADIUS)

    def draw_board_grid_lines(self):
        """单独绘制网格线，确保它们在最上层"""
        for i in range(self.n + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (MARGIN + i * CELL_SIZE, MARGIN),
                             (MARGIN + i * CELL_SIZE, WINDOW_SIZE - MARGIN))
            pygame.draw.line(self.screen, GRID_COLOR, (MARGIN, MARGIN + i * CELL_SIZE),
                             (WINDOW_SIZE - MARGIN, MARGIN + i * CELL_SIZE))

    def display_game_info(self):
        """在窗口顶部显示当前玩家和游戏状态"""
        if self.game_over:
            if self.winner == 0:
                text = "游戏结束：平局"
            else:
                winner_name = "黑棋 (红方)" if self.winner == 1 else "白棋 (蓝方)"
                text = f"游戏结束：{winner_name} 获胜！"
        else:
            player_name = "黑棋 (红方)" if self.current_player == 1 else "白棋 (蓝方)"
            text = f"当前玩家: {player_name}"

        text_surface = self.font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(WINDOW_SIZE // 2, MARGIN // 2))
        self.screen.blit(text_surface, text_rect)

    def handle_click(self, pos):
        """处理鼠标点击事件，进行落子"""
        if self.game_over:
            return

        x, y = pos
        # 将屏幕坐标转换为棋盘网格坐标
        if MARGIN < x < WINDOW_SIZE - MARGIN and MARGIN < y < WINDOW_SIZE - MARGIN:
            r = (y - MARGIN) // CELL_SIZE
            c = (x - MARGIN) // CELL_SIZE

            action = r * self.n + c

            # 从 C++ 核心获取合法走法
            valid_moves = self.game.getValidMoves(self.board, self.current_player)

            if valid_moves[action] == 1:
                # 如果是合法走法，则更新游戏状态
                # FIX: 将返回的 list 转换为 numpy array
                next_board_list, self.current_player, self.hash = self.game.getNextState(self.board,
                                                                                         self.current_player, action)
                self.board = np.array(next_board_list, dtype=np.float32)

                # 检查游戏是否结束
                # 注意：getGameEnded 的第二个参数应该是原来的玩家
                self.winner = self.game.getGameEnded(self.board, self.current_player * -1)
                if self.winner != 0:
                    self.game_over = True
            else:
                print(f"无效落子位置: ({r}, {c})")

    def run(self):
        """主游戏循环"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            self.draw_board()
            self.draw_pieces_and_territory()
            self.display_game_info()

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    visualizer = GameVisualizer()
    visualizer.run()
