# play_vs_ai.py
# 运行此文件以启动具有高级功能的人机对战GUI。

import pygame
import numpy as np
import sys
import os
import torch
import math
import threading  # <-- 1. 引入 threading 模块
import katago_cpp_core
from nn_wrapper import NNetWrapper
from nn_model import NNet

# --- 颜色定义 ---
C_BLACK = (0, 0, 0)
C_WHITE = (255, 255, 255)
C_GRID = (50, 50, 50)
C_PLAYER_1 = (220, 50, 50)  # 红色 (P1)
C_PLAYER_2 = (50, 100, 220)  # 蓝色 (P2)
C_P1_CONTROL = (255, 224, 224)  # 淡红色
C_P2_CONTROL = (224, 224, 255)  # 浅蓝色
C_BUTTON = (100, 100, 100)
C_BUTTON_HOVER = (130, 130, 130)
C_BUTTON_TEXT = (255, 255, 255)
C_INFO_TEXT = (30, 30, 30)

BOARD_SIZE = 9
MAX_ROUNDS = 50
CELL_SIZE = 60
MARGIN = 40
BOARD_DIM = BOARD_SIZE * CELL_SIZE
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_DIM + MARGIN * 2 + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_DIM + MARGIN * 2
PIECE_RADIUS = CELL_SIZE // 2 - 4


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'num_channels': 13,
    'num_filters': 128,
    'num_residual_blocks': 8,
    'numMCTSSims': 800,
    'cpuct': 1.0,
    'epsilon': 0,
    'checkpoint': './checkpoint/',
    'load_model_file': 'best.pth.tar',
})


class Button:

    def __init__(self, rect, text, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, screen, font):
        color = C_BUTTON_HOVER if self.rect.collidepoint(pygame.mouse.get_pos()) else C_BUTTON
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        text_surf = font.render(self.text, True, C_BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)


class GameGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("人机对战 & 分析工具")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font_info = pygame.font.SysFont("simhei", 24)
        self.font_button = pygame.font.SysFont("simhei", 20)
        self.font_analysis = pygame.font.SysFont("arial", 12)

        self.game_state = 'MENU'
        self.game_mode = None

        self.analysis_data = None
        self.show_analysis = False

        # --- 2. 增加AI状态变量 ---
        self.is_ai_thinking = False
        self.ai_move_result = None
        self.ai_thread = None

        self.clock = pygame.time.Clock()
        self._initialize_neural_net_and_mcts()
        self._create_buttons()

    def _initialize_neural_net_and_mcts(self):
        self.game_core = katago_cpp_core.Game(BOARD_SIZE, MAX_ROUNDS)
        try:
            self.nnet = NNetWrapper(self.game_core, NNet, args)
            self.nnet.load_checkpoint(args.checkpoint, args.load_model_file)
            print("神经网络模型加载成功。")
        except FileNotFoundError:
            print(f"错误：找不到模型文件 {os.path.join(args.checkpoint, args.load_model_file)}")
            sys.exit(1)

        mcts_args = katago_cpp_core.MCTSArgs()
        mcts_args.numMCTSSims = args.numMCTSSims
        mcts_args.cpuct = args.cpuct
        mcts_args.epsilon = args.epsilon
        self.mcts = katago_cpp_core.MCTS(self.game_core, self.nnet.predict_batch, mcts_args)
        print("MCTS引擎初始化成功。")

    def _create_buttons(self):
        panel_x = BOARD_DIM + MARGIN * 2
        self.menu_buttons = [
            Button((panel_x + 50, 150, 200, 50), "玩家 vs 玩家", lambda: self.start_game('P_vs_P')),
            Button((panel_x + 50, 220, 200, 50), "执黑 vs AI", lambda: self.start_game('P_vs_AI_Black')),
            Button((panel_x + 50, 290, 200, 50), "执白 vs AI", lambda: self.start_game('P_vs_AI_White')),
            Button((panel_x + 50, 360, 200, 50), "AI vs AI", lambda: self.start_game('AI_vs_AI')),
        ]
        self.game_buttons = [
            Button((panel_x + 50, WINDOW_HEIGHT - 180, 200, 40), "切换分析显示", self.toggle_analysis),
            Button((panel_x + 50, WINDOW_HEIGHT - 120, 200, 40), "返回菜单", self.go_to_menu)
        ]

    def start_game(self, mode):
        self.game_mode = mode
        self.board, self.hash = self.game_core.getInitialBoard()
        self.board = np.array(self.board, dtype=np.float32)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.analysis_data = None
        self.game_state = 'PLAYING'
        # 重置AI状态
        self.is_ai_thinking = False
        self.ai_move_result = None
        print(f"游戏开始，模式: {mode}")

    def go_to_menu(self):
        self.game_state = 'MENU'
        # 离开游戏时重置AI状态，防止线程结果影响菜单
        self.is_ai_thinking = False
        self.ai_move_result = None

    def toggle_analysis(self):
        self.show_analysis = not self.show_analysis
        if self.show_analysis:
            self.run_analysis()
        else:
            self.analysis_data = None

    def run_analysis(self):
        if self.game_over:
            self.analysis_data = None
            return
        canonical_board, _ = self.game_core.getCanonicalForm(self.board, self.hash, self.current_player)
        canonical_board_tensor = torch.FloatTensor(canonical_board).contiguous().to(self.nnet.device)
        batch = canonical_board_tensor.unsqueeze(0)
        batch = batch.cpu()
        pi, v, score, score_var, ownership = self.nnet.predict_batch(batch)
        self.analysis_data = {
            'pi': pi[0], 'v': (v[0][0] + 1) / 2, 'score': score[0][0],
            'score_var': score_var[0][0], 'ownership': ownership[0][0]
        }
        if self.current_player == -1:
            self.analysis_data['ownership'] *= -1

    def run(self):
        while True:
            self.handle_events()
            if self.game_state == 'PLAYING':
                self.update_game()
            self.draw()
            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            active_buttons = self.menu_buttons if self.game_state == 'MENU' else self.game_buttons
            for btn in active_buttons:
                btn.handle_event(event)

            is_human_turn = (self.game_mode == 'P_vs_P') or \
                            (self.game_mode == 'P_vs_AI_Black' and self.current_player == 1) or \
                            (self.game_mode == 'P_vs_AI_White' and self.current_player == -1)

            if self.game_state == 'PLAYING' and is_human_turn and not self.is_ai_thinking:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_board_click(event.pos)

    def update_game(self):
        """处理游戏逻辑更新，例如AI移动"""
        if self.game_over: return

        # --- 4. 修改主更新循环 ---
        # 检查AI是否完成了思考
        if self.ai_move_result is not None:
            action = self.ai_move_result
            self.perform_move(action)
            # 重置状态，为AI的下一回合做准备
            self.ai_move_result = None
            self.is_ai_thinking = False

        is_ai_turn = (self.game_mode == 'AI_vs_AI') or \
                     (self.game_mode == 'P_vs_AI_Black' and self.current_player == -1) or \
                     (self.game_mode == 'P_vs_AI_White' and self.current_player == 1)

        # 如果轮到AI下棋，并且它当前没有在思考，则启动思考线程
        if is_ai_turn and not self.is_ai_thinking:
            self.ai_move()

    def draw(self):
        self.screen.fill(C_WHITE)
        if self.game_state == 'MENU':
            self.draw_menu()
        elif self.game_state in ['PLAYING', 'GAME_OVER']:
            self.draw_game_screen()
        pygame.display.flip()

    def draw_menu(self):
        title_surf = self.font_info.render("选择游戏模式", True, C_BLACK)
        title_rect = title_surf.get_rect(center=(WINDOW_WIDTH / 2, 100))
        self.screen.blit(title_surf, title_rect)
        for btn in self.menu_buttons:
            btn.draw(self.screen, self.font_button)

    def draw_game_screen(self):
        self.draw_board_and_pieces()
        if self.show_analysis and self.analysis_data:
            self.draw_analysis_overlay()
        self.draw_side_panel()

    def draw_board_and_pieces(self):
        # 绘制背景和网格线
        board_reshaped = self.board.reshape(13, BOARD_SIZE, BOARD_SIZE)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = pygame.Rect(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if board_reshaped[2, r, c] == 1.0:
                    pygame.draw.rect(self.screen, C_P1_CONTROL, rect)
                elif board_reshaped[3, r, c] == 1.0:
                    pygame.draw.rect(self.screen, C_P2_CONTROL, rect)

        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(self.screen, C_GRID, (MARGIN + i * CELL_SIZE, MARGIN),
                             (MARGIN + i * CELL_SIZE, MARGIN + BOARD_DIM))
            pygame.draw.line(self.screen, C_GRID, (MARGIN, MARGIN + i * CELL_SIZE),
                             (MARGIN + BOARD_DIM, MARGIN + i * CELL_SIZE))

        # 绘制棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                center = (MARGIN + c * CELL_SIZE + CELL_SIZE // 2, MARGIN + r * CELL_SIZE + CELL_SIZE // 2)
                if board_reshaped[0, r, c] == 1.0:
                    pygame.draw.circle(self.screen, C_PLAYER_1, center, PIECE_RADIUS)
                elif board_reshaped[1, r, c] == 1.0:
                    pygame.draw.circle(self.screen, C_PLAYER_2, center, PIECE_RADIUS)

    def draw_analysis_overlay(self):
        if not self.analysis_data: return
        overlay = pygame.Surface((BOARD_DIM, BOARD_DIM), pygame.SRCALPHA)
        max_pi = np.max(self.analysis_data['pi'])
        if max_pi > 0:
            for i, prob in enumerate(self.analysis_data['pi']):
                if prob > 0.01:
                    r, c = i // BOARD_SIZE, i % BOARD_SIZE
                    alpha = int(200 * (prob / max_pi))
                    color = (34, 139, 34, alpha)
                    pygame.draw.rect(overlay, color, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        ownership_map = self.analysis_data['ownership']
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                own_val = ownership_map[r, c]
                if abs(own_val) > 0.1:
                    alpha = int(100 * abs(own_val))
                    color = C_PLAYER_1 if own_val > 0 else C_PLAYER_2
                    rect_size = int(CELL_SIZE * 0.5)
                    rect_pos = (
                        c * CELL_SIZE + (CELL_SIZE - rect_size) / 2, r * CELL_SIZE + (CELL_SIZE - rect_size) / 2)
                    own_rect = pygame.Rect(rect_pos[0], rect_pos[1], rect_size, rect_size)
                    pygame.draw.rect(overlay, (*color, alpha), own_rect, border_radius=4)
        self.screen.blit(overlay, (MARGIN, MARGIN))

    def draw_side_panel(self):
        panel_x = BOARD_DIM + MARGIN * 2
        title_text = "游戏状态" if not self.game_over else "游戏结束"
        title_surf = self.font_info.render(title_text, True, C_BLACK)
        self.screen.blit(title_surf, (panel_x + 20, 20))

        board_reshaped = self.board.reshape(13, BOARD_SIZE, BOARD_SIZE)
        round_num = int(board_reshaped[4, 0, 0] * MAX_ROUNDS)
        p1_control = int(np.sum(board_reshaped[2]))
        p2_control = int(np.sum(board_reshaped[3]))
        info_texts = [f"回合: {round_num} / {MAX_ROUNDS}", f"红方控制区: {p1_control}", f"蓝方控制区: {p2_control}"]

        if self.game_over:
            winner_text = "平局"
            if self.winner != 0:
                winner_name = "红方" if self.winner == 1 else "蓝方"
                winner_text = f"胜者: {winner_name}"
            info_texts.append(winner_text)
        else:
            player_name = "红方" if self.current_player == 1 else "蓝方"
            is_human_turn = (self.game_mode == 'P_vs_P') or \
                            (self.game_mode == 'P_vs_AI_Black' and self.current_player == 1) or \
                            (self.game_mode == 'P_vs_AI_White' and self.current_player == -1)

            if is_human_turn:
                turn_text = "当前玩家: 你"
            else:
                turn_text = "当前玩家: AI 思考中..."
            info_texts.append(turn_text)

        for i, text in enumerate(info_texts):
            surf = self.font_info.render(text, True, C_INFO_TEXT)
            self.screen.blit(surf, (panel_x + 20, 60 + i * 35))

        if self.show_analysis and self.analysis_data:
            analysis_texts = [
                "--- AI 分析 ---", f"胜率 (Value): {self.analysis_data['v']:.2%}",
                f"预测得分 (Score): {self.analysis_data['score']:.2f} ± {math.sqrt(self.analysis_data['score_var']):.2f}",
            ]
            for i, text in enumerate(analysis_texts):
                surf = self.font_info.render(text, True, C_INFO_TEXT)
                self.screen.blit(surf, (panel_x + 20, 220 + i * 35))

        for btn in self.game_buttons:
            btn.draw(self.screen, self.font_button)

    def handle_board_click(self, pos):
        if self.game_over: return
        board_x = pos[0] - MARGIN
        board_y = pos[1] - MARGIN
        if 0 <= board_x < BOARD_DIM and 0 <= board_y < BOARD_DIM:
            r, c = board_y // CELL_SIZE, board_x // CELL_SIZE
            action = r * BOARD_SIZE + c
            valid_moves = self.game_core.getValidMoves(self.board, self.current_player)
            if valid_moves[action] == 1:
                self.perform_move(action)
            else:
                print(f"无效落子位置: ({r}, {c})")

    # --- 3. 重构AI移动逻辑 ---
    def _ai_move_task(self):
        """这个函数在子线程中运行，执行耗时的MCTS计算。"""
        canonical_board, canonical_hash = self.game_core.getCanonicalForm(self.board, self.hash, self.current_player)
        seeds = np.random.randint(0, 2 ** 32 - 1, size=1, dtype=np.uint32)

        # 这是阻塞操作，但在子线程中，所以不会冻结UI
        pi = self.mcts.getActionProbs([canonical_board], [canonical_hash], seeds.tolist(), temp=0)[0]

        action = np.argmax(pi)

        # 将结果存入实例变量，以便主线程可以获取
        self.ai_move_result = action

    def ai_move(self):
        """启动AI思考的子线程。"""
        self.is_ai_thinking = True
        self.ai_move_result = None  # 清除上一回合的结果

        # 创建并启动子线程
        self.ai_thread = threading.Thread(target=self._ai_move_task)
        self.ai_thread.daemon = True  # 确保主程序退出时子线程也会退出
        self.ai_thread.start()

    def perform_move(self, action):
        """执行一个落子动作并更新游戏状态。"""
        next_board_list, next_player, next_hash = self.game_core.getNextState(self.board, self.current_player, action)
        self.board = np.array(next_board_list, dtype=np.float32)
        self.current_player = next_player
        self.hash = next_hash

        self.winner = self.game_core.getGameEnded(self.board, self.current_player )
        if self.winner != 0:
            self.game_over = True
            self.game_state = 'GAME_OVER'

        if self.show_analysis:
            self.run_analysis()


if __name__ == '__main__':
    gui = GameGUI()
    gui.run()
