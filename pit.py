# play_vs_ai.py
# 运行此文件以启动具有高级功能的人机对战GUI。

import pygame
import numpy as np
import sys
import os
import torch
import math
import threading
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
    'numMCTSSims': 800,  # AI落子时的模拟次数
    'numMCTSSimsAnal': 200,  # 分析功能使用的模拟次数
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
        self.game_over = True  # 初始化 game_over 状态

        self.analysis_data = None
        self.show_analysis = False

        # --- AI 移动线程相关状态 ---
        self.is_ai_thinking = False
        self.ai_move_result = None

        # --- MCTS 分析线程相关状态 ---
        self.is_analysis_running = False
        self.analysis_result = None
        self.analysis_thread = None

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

        # MCTS for AI's actual moves
        mcts_args_move = katago_cpp_core.MCTSArgs()
        mcts_args_move.numMCTSSims = args.numMCTSSims
        mcts_args_move.cpuct = args.cpuct
        mcts_args_move.epsilon = args.epsilon
        self.mcts = katago_cpp_core.MCTS(self.game_core, self.nnet.predict_batch, mcts_args_move)
        print(f"AI移动MCTS引擎初始化成功 (模拟次数: {args.numMCTSSims})。")

        # A separate MCTS instance for the analysis feature
        mcts_args_anal = katago_cpp_core.MCTSArgs()
        mcts_args_anal.numMCTSSims = args.numMCTSSimsAnal
        mcts_args_anal.cpuct = args.cpuct
        mcts_args_anal.epsilon = args.epsilon
        self.mcts_analysis = katago_cpp_core.MCTS(self.game_core, self.nnet.predict_batch, mcts_args_anal)
        print(f"局面分析MCTS引擎初始化成功 (模拟次数: {args.numMCTSSimsAnal})。")

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
        self.is_ai_thinking = False
        self.ai_move_result = None
        self.is_analysis_running = False
        self.analysis_result = None
        print(f"游戏开始，模式: {mode}")
        if self.show_analysis:
            self.run_deep_analysis()

    def go_to_menu(self):
        self.game_state = 'MENU'
        self.is_ai_thinking = False
        self.is_analysis_running = False
        self.game_over = True  # 返回菜单时，将游戏标记为结束

    def toggle_analysis(self):
        self.show_analysis = not self.show_analysis
        if self.show_analysis and self.game_state == 'PLAYING':
            self.run_deep_analysis()
        else:
            self.analysis_data = None
            self.is_analysis_running = False

    def _deep_analysis_task(self):
        """在后台线程中运行MCTS分析"""
        print(f"为玩家 {self.current_player} 运行当前局面的深度分析 (模拟次数: {args.numMCTSSimsAnal})...")
        # 1. 从MCTS获取策略向量
        canonical_board, canonical_hash = self.game_core.getCanonicalForm(self.board, self.hash, self.current_player)
        seeds = np.random.randint(0, 2 ** 32 - 1, size=1, dtype=np.uint32)
        # --- 使用专用的分析MCTS实例 ---
        pi = self.mcts_analysis.getActionProbs([canonical_board], [canonical_hash], seeds.tolist(), temp=1)[0]

        # 2. 从神经网络获取价值、得分等信息
        canonical_board_tensor = torch.FloatTensor(canonical_board).contiguous().to(self.nnet.device)
        batch = canonical_board_tensor.unsqueeze(0).cpu()
        _, v, score, score_var, ownership = self.nnet.predict_batch(batch)

        # 3. 组合成最终的分析数据
        analysis_dict = {
            'pi': pi,  # 使用MCTS的策略
            'v': (v[0][0] + 1) / 2,
            'score': score[0][0],
            'score_var': score_var[0][0],
            'ownership': ownership[0][0]
        }
        if self.current_player == -1:
            analysis_dict['ownership'] *= -1

        self.analysis_result = analysis_dict

    def run_deep_analysis(self):
        """启动MCTS深度分析线程"""
        if self.is_analysis_running or self.game_over or self.game_state != 'PLAYING':
            return
        # --- 修复: 立即清除旧的分析数据以防止残留显示 ---
        self.analysis_data = None
        self.is_analysis_running = True
        self.analysis_result = None
        self.analysis_thread = threading.Thread(target=self._deep_analysis_task)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def run(self):
        while True:
            self.handle_events()
            # --- 修复: 只有在游戏进行中才更新游戏逻辑 ---
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
        if self.game_over: return

        # 检查AI移动线程是否完成
        if self.ai_move_result is not None:
            action = self.ai_move_result
            self.perform_move(action)
            self.ai_move_result = None
            self.is_ai_thinking = False

        # 检查分析线程是否完成
        if self.analysis_result is not None:
            self.analysis_data = self.analysis_result
            self.analysis_result = None
            self.is_analysis_running = False
            print("深度分析完成，已更新界面。")

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

        pi_values = self.analysis_data['pi']
        if not isinstance(pi_values, np.ndarray):
            pi_values = np.array(pi_values)

        max_pi = np.max(pi_values)
        if max_pi > 0:
            for i, prob in enumerate(pi_values):
                if prob > 0.001:
                    r, c = i // BOARD_SIZE, i % BOARD_SIZE
                    alpha = int(220 * (prob / max_pi))
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
            turn_text = f"当前玩家: {player_name}"
            if self.is_ai_thinking:
                turn_text = "AI 思考中..."
            info_texts.append(turn_text)

        for i, text in enumerate(info_texts):
            surf = self.font_info.render(text, True, C_INFO_TEXT)
            self.screen.blit(surf, (panel_x + 20, 60 + i * 35))

        if self.show_analysis:
            if self.is_analysis_running:
                analysis_texts = ["--- AI 分析 ---", "分析中..."]
            elif self.analysis_data:
                analysis_texts = [
                    "--- AI 深度分析 ---",
                    f"胜率 (Value): {self.analysis_data['v']:.2%}",
                    f"预测得分 (Score): {self.analysis_data['score']:.2f} ± {math.sqrt(self.analysis_data['score_var']):.2f}",
                ]
            else:
                analysis_texts = ["--- AI 分析 ---", "等待分析..."]

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

    def _ai_move_task(self):
        canonical_board, canonical_hash = self.game_core.getCanonicalForm(self.board, self.hash, self.current_player)
        seeds = np.random.randint(0, 2 ** 32 - 1, size=1, dtype=np.uint32)
        pi = self.mcts.getActionProbs([canonical_board], [canonical_hash], seeds.tolist(), temp=0)[0]
        action = np.argmax(pi)
        self.ai_move_result = action

    def ai_move(self):
        if self.is_ai_thinking: return
        self.is_ai_thinking = True
        self.ai_move_result = None
        ai_thread = threading.Thread(target=self._ai_move_task)
        ai_thread.daemon = True
        ai_thread.start()

    def perform_move(self, action):
        next_board_list, next_player, next_hash = self.game_core.getNextState(self.board, self.current_player, action)
        self.board = np.array(next_board_list, dtype=np.float32)
        self.current_player = next_player
        self.hash = next_hash
        self.winner = self.game_core.getGameEnded(self.board, self.current_player)
        if self.winner != 0:
            self.game_over = True
            self.game_state = 'GAME_OVER'
            self.is_analysis_running = False  # 游戏结束，停止分析

        # 移动后，如果分析是开启的，就为新局面重新运行深度分析
        if self.show_analysis and not self.game_over:
            self.run_deep_analysis()


if __name__ == '__main__':
    gui = GameGUI()
    gui.run()
