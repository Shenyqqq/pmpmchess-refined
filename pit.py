# pit.py
# 运行此文件以启动具有悔棋和动态AI参数调整功能的高级GUI。

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
C_PLAYER_1 = (220, 50, 50)
C_PLAYER_2 = (50, 100, 220)
C_P1_CONTROL = (255, 224, 224)
C_P2_CONTROL = (224, 224, 255)
C_BUTTON = (100, 100, 100)
C_BUTTON_HOVER = (130, 130, 130)
C_BUTTON_TEXT = (255, 255, 255)
C_INFO_TEXT = (30, 30, 30)
C_INPUT_BOX_ACTIVE = (150, 150, 200)
C_INPUT_BOX_INACTIVE = (200, 200, 200)

# --- 布局和游戏参数 ---
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

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.callback()
            return True
        return False

    def draw(self, screen, font):
        color = C_BUTTON_HOVER if self.rect.collidepoint(pygame.mouse.get_pos()) else C_BUTTON
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        text_surf = font.render(self.text, True, C_BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)


class TextInputBox:
    """一个简单的文本输入框类，支持整数和小数"""

    def __init__(self, rect, initial_text="", allow_float=False):
        self.rect = pygame.Rect(rect)
        self.text = initial_text
        self.active = False
        self.allow_float = allow_float

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text += event.unicode
            elif self.allow_float and event.unicode == '.' and '.' not in self.text:
                self.text += event.unicode

    def draw(self, screen, font):
        color = C_INPUT_BOX_ACTIVE if self.active else C_INPUT_BOX_INACTIVE
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        text_surf = font.render(self.text, True, C_BLACK)
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))


class GameGUI:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("泡姆棋")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font_info = pygame.font.SysFont("simhei", 24)
        self.font_button = pygame.font.SysFont("simhei", 20)
        self.font_label = pygame.font.SysFont("simhei", 18)

        self.game_state = 'MENU'
        self.game_mode = None
        self.game_over = True
        self.game_history = []

        self.analysis_data = None
        self.show_analysis = False
        self.is_ai_thinking = False
        self.ai_move_result = None
        self.is_analysis_running = False
        self.analysis_result = None
        self.analysis_thread = None

        self.clock = pygame.time.Clock()
        self._initialize_neural_net()
        self._create_ui_elements()

    def _initialize_neural_net(self):
        self.game_core = katago_cpp_core.Game(BOARD_SIZE, MAX_ROUNDS)
        try:
            self.nnet = NNetWrapper(self.game_core, NNet, args)
            self.nnet.load_checkpoint(args.checkpoint, args.load_model_file)
            print("神经网络模型加载成功。")
        except FileNotFoundError:
            print(f"错误：找不到模型文件 {os.path.join(args.checkpoint, args.load_model_file)}")
            sys.exit(1)

    def _create_ui_elements(self):
        panel_x = BOARD_DIM + MARGIN * 2
        self.menu_buttons = [
            Button((panel_x + 50, 150, 200, 50), "玩家 vs 玩家", lambda: self.start_game('P_vs_P')),
            Button((panel_x + 50, 220, 200, 50), "执黑 vs AI", lambda: self.start_game('P_vs_AI_Black')),
            Button((panel_x + 50, 290, 200, 50), "执白 vs AI", lambda: self.start_game('P_vs_AI_White')),
            Button((panel_x + 50, 360, 200, 50), "AI vs AI", lambda: self.start_game('AI_vs_AI')),
        ]

        # 修复UI重叠：调整按钮和输入框的Y坐标
        self.game_buttons = [
            Button((panel_x + 50, 320, 200, 40), "悔棋", self.undo_move),
            Button((panel_x + 50, 370, 200, 40), "切换分析显示", self.toggle_analysis),
            Button((panel_x + 50, 420, 200, 40), "返回菜单", self.go_to_menu)
        ]

        self.sims_move_input = TextInputBox((panel_x + 160, 490, 90, 30), "2000")
        self.sims_anal_input = TextInputBox((panel_x + 160, 530, 90, 30), "200")
        self.temp_input = TextInputBox((panel_x + 160, 570, 90, 30), "1.0", allow_float=True)  # 允许输入小数
        self.input_boxes = [self.sims_move_input, self.sims_anal_input, self.temp_input]

    def start_game(self, mode):
        self.game_mode = mode
        self.board, self.hash = self.game_core.getInitialBoard()
        self.board = np.array(self.board, dtype=np.float32)
        self.current_player = 1
        self.game_over = False
        self.game_history = []
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
        self.game_over = True

    def undo_move(self):
        """智能悔棋功能"""
        if self.is_ai_thinking:
            print("无法在AI思考时悔棋。")
            return

        # 决定悔棋步数
        undo_steps = 1
        is_p_vs_ai = self.game_mode in ['P_vs_AI_Black', 'P_vs_AI_White']
        is_human_turn_now = (self.game_mode == 'P_vs_AI_Black' and self.current_player == 1) or \
                            (self.game_mode == 'P_vs_AI_White' and self.current_player == -1)

        if is_p_vs_ai and is_human_turn_now and len(self.game_history) >= 2:
            undo_steps = 2

        if len(self.game_history) < undo_steps:
            print(f"历史记录不足，无法悔棋 {undo_steps} 步。")
            return

        print(f"执行悔棋 {undo_steps} 步...")

        # 弹出相应步数
        for _ in range(undo_steps):
            self.game_history.pop()

        # 恢复到悔棋后的最新状态
        if self.game_history:
            last_state = self.game_history[-1]
            prev_state = self.game_history.pop()  # 再次弹出以获取该状态，之后再添加回去
            self.board = prev_state['board']
            self.hash = prev_state['hash']
            self.current_player = prev_state['player']
            # 重新执行这一步，以确保历史记录的连续性
            self.perform_move(prev_state['last_action'])

        else:  # 如果悔棋到开局
            self.start_game(self.game_mode)
            # 为了避免调用两次run_deep_analysis，我们在这里直接返回
            return

        print("悔棋成功。")
        if self.show_analysis:
            self.run_deep_analysis()
        else:
            self.analysis_data = None

    def toggle_analysis(self):
        self.show_analysis = not self.show_analysis
        if self.show_analysis and self.game_state == 'PLAYING':
            self.run_deep_analysis()
        else:
            self.analysis_data = None
            self.is_analysis_running = False

    def _create_mcts_instance(self, sims, temp=1.0):
        mcts_args = katago_cpp_core.MCTSArgs()
        try:
            mcts_args.numMCTSSims = int(sims) if sims else 100
        except (ValueError, TypeError):
            mcts_args.numMCTSSims = 100
        mcts_args.cpuct = args.cpuct
        mcts_args.epsilon = args.epsilon
        return katago_cpp_core.MCTS(self.game_core, self.nnet.predict_batch, mcts_args)

    def _deep_analysis_task(self):
        sims = self.sims_anal_input.text
        mcts_anal = self._create_mcts_instance(sims)
        canonical_board, canonical_hash = self.game_core.getCanonicalForm(self.board, self.hash, self.current_player)
        seeds = np.random.randint(0, 2 ** 32 - 1, size=1, dtype=np.uint32)
        pi = mcts_anal.getActionProbs([canonical_board], [canonical_hash], seeds.tolist(), temp=1)[0]
        canonical_board_tensor = torch.FloatTensor(canonical_board).contiguous().to(self.nnet.device)
        batch = canonical_board_tensor.unsqueeze(0).cpu()
        _, v, score, score_var, ownership = self.nnet.predict_batch(batch)
        analysis_dict = {
            'pi': pi, 'v': (v[0][0] + 1) / 2, 'score': score[0][0],
            'score_var': score_var[0][0], 'ownership': ownership[0][0]
        }
        if self.current_player == -1:
            analysis_dict['ownership'] *= -1
        self.analysis_result = analysis_dict

    def run_deep_analysis(self):
        if self.is_analysis_running or self.game_over or self.game_state != 'PLAYING':
            return
        self.analysis_data = None
        self.is_analysis_running = True
        self.analysis_result = None
        self.analysis_thread = threading.Thread(target=self._deep_analysis_task, daemon=True)
        self.analysis_thread.start()

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
                pygame.quit(), sys.exit()
            active_buttons = self.menu_buttons if self.game_state == 'MENU' else self.game_buttons
            for btn in active_buttons:
                btn.handle_event(event)
            for box in self.input_boxes:
                box.handle_event(event)
            is_human_turn = (self.game_mode == 'P_vs_P') or \
                            (self.game_mode == 'P_vs_AI_Black' and self.current_player == 1) or \
                            (self.game_mode == 'P_vs_AI_White' and self.current_player == -1)
            if self.game_state == 'PLAYING' and is_human_turn and not self.is_ai_thinking:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_on_ui = any(btn.rect.collidepoint(event.pos) for btn in self.game_buttons) or \
                                    any(box.rect.collidepoint(event.pos) for box in self.input_boxes)
                    if not clicked_on_ui:
                        self.handle_board_click(event.pos)

    def update_game(self):
        if self.game_over: return
        if self.ai_move_result is not None:
            self.perform_move(self.ai_move_result)
            self.ai_move_result = None
            self.is_ai_thinking = False
        if self.analysis_result is not None:
            self.analysis_data = self.analysis_result
            self.analysis_result = None
            self.is_analysis_running = False
        is_ai_turn = (self.game_mode == 'AI_vs_AI') or \
                     (self.game_mode == 'P_vs_AI_Black' and self.current_player == -1) or \
                     (self.game_mode == 'P_vs_AI_White' and self.current_player == 1)
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
        for btn in self.menu_buttons: btn.draw(self.screen, self.font_button)

    def draw_game_screen(self):
        self.draw_board_and_pieces()
        if self.show_analysis and self.analysis_data: self.draw_analysis_overlay()
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
        pi_values = np.array(self.analysis_data['pi'])
        max_pi = np.max(pi_values) if len(pi_values) > 0 else 0
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
        info_texts = [f"回合: {round_num}/{MAX_ROUNDS}", f"红方控制区: {p1_control}", f"蓝方控制区: {p2_control}"]
        if self.game_over:
            winner_text = "平局"
            if self.winner != 0: winner_text = f"胜者: {'红方' if self.winner == 1 else '蓝方'}"
            info_texts.append(winner_text)
        else:
            turn_text = f"当前玩家: {'红方' if self.current_player == 1 else '蓝方'}"
            if self.is_ai_thinking: turn_text = "AI 思考中..."
            info_texts.append(turn_text)
        for i, text in enumerate(info_texts):
            self.screen.blit(self.font_info.render(text, True, C_INFO_TEXT), (panel_x + 20, 60 + i * 35))
        if self.show_analysis:
            analysis_title = "--- AI 深度分析 ---"
            if self.is_analysis_running:
                analysis_texts = ["分析中..."]
            elif self.analysis_data:
                analysis_texts = [f"胜率: {self.analysis_data['v']:.2%}",
                                  f"得分: {self.analysis_data['score']:.2f} ± {math.sqrt(self.analysis_data['score_var']):.2f}"]
            else:
                analysis_texts = ["等待分析..."]
            self.screen.blit(self.font_info.render(analysis_title, True, C_INFO_TEXT), (panel_x + 20, 200))
            for i, text in enumerate(analysis_texts):
                self.screen.blit(self.font_info.render(text, True, C_INFO_TEXT), (panel_x + 20, 235 + i * 35))

        # 修复UI重叠：调整AI参数和按钮的Y坐标
        y_offset = 460
        self.screen.blit(self.font_info.render("--- AI 参数设置 ---", True, C_INFO_TEXT), (panel_x + 20, y_offset))
        y_offset += 35
        self.screen.blit(self.font_label.render("模拟次数 (行棋)", True, C_INFO_TEXT), (panel_x + 20, y_offset + 5))
        self.sims_move_input.rect.y = y_offset
        self.sims_move_input.draw(self.screen, self.font_label)
        y_offset += 40
        self.screen.blit(self.font_label.render("温度 (行棋)", True, C_INFO_TEXT), (panel_x + 20, y_offset + 5))
        self.temp_input.rect.y = y_offset
        self.temp_input.draw(self.screen, self.font_label)
        y_offset += 40
        self.screen.blit(self.font_label.render("模拟次数 (分析)", True, C_INFO_TEXT), (panel_x + 20, y_offset + 5))
        self.sims_anal_input.rect.y = y_offset
        self.sims_anal_input.draw(self.screen, self.font_label)



        for btn in self.game_buttons: btn.draw(self.screen, self.font_button)

    def handle_board_click(self, pos):
        if self.game_over: return
        board_x, board_y = pos[0] - MARGIN, pos[1] - MARGIN
        if 0 <= board_x < BOARD_DIM and 0 <= board_y < BOARD_DIM:
            r, c = board_y // CELL_SIZE, board_x // CELL_SIZE
            action = r * BOARD_SIZE + c
            if self.game_core.getValidMoves(self.board, self.current_player)[action] == 1:
                self.perform_move(action)
            else:
                print(f"无效落子位置: ({r}, {c})")

    def _ai_move_task(self):
        sims = self.sims_move_input.text
        temp_str = self.temp_input.text
        try:
            temp = float(temp_str) if temp_str else 1.0
        except (ValueError, TypeError):
            temp = 1.0
        mcts_move = self._create_mcts_instance(sims)
        canonical_board, canonical_hash = self.game_core.getCanonicalForm(self.board, self.hash, self.current_player)
        seeds = np.random.randint(0, 2 ** 32 - 1, size=1, dtype=np.uint32)
        pi = mcts_move.getActionProbs([canonical_board], [canonical_hash], seeds.tolist(), temp=temp)[0]
        self.ai_move_result = np.argmax(pi)

    def ai_move(self):
        if self.is_ai_thinking: return
        self.is_ai_thinking = True
        self.ai_move_result = None
        threading.Thread(target=self._ai_move_task, daemon=True).start()

    def perform_move(self, action):
        # 保存当前状态到历史记录，用于悔棋
        self.game_history.append({
            'board': self.board.copy(),
            'hash': self.hash,
            'player': self.current_player,
            'last_action': action  # 保存上一步的动作，用于悔棋后恢复
        })
        next_board_list, next_player, next_hash = self.game_core.getNextState(self.board, self.current_player, action)
        self.board = np.array(next_board_list, dtype=np.float32)
        self.current_player = next_player
        self.hash = next_hash
        self.winner = self.game_core.getGameEnded(self.board, self.current_player)
        if self.winner != 0:
            self.game_over = True
            self.game_state = 'GAME_OVER'
            self.is_analysis_running = False
        if self.show_analysis and not self.game_over:
            self.run_deep_analysis()


if __name__ == '__main__':
    gui = GameGUI()
    gui.run()
