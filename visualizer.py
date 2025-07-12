import pygame
import sys
from typing import List, Tuple, Set

# Initialize pygame
pygame.init()

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_RED = (255, 200, 200)  # Light Red for player 1 controlled area
LIGHT_BLUE = (200, 200, 255) # Light Blue for player -1 controlled area
GRAY = (200, 200, 200)

font_path = "C:\Windows\Fonts\等线.ttf" # This path might need to be adjusted based on the system

class GameVisualizer:
    def __init__(self, game, cell_size=60, padding=20):
        self.game = game
        self.cell_size = cell_size
        self.padding = padding
        self.radius = cell_size // 3

        self.width = game.size * cell_size + 2 * padding
        self.height = game.size * cell_size + 2 * padding

        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("连三棋")
        # Attempt to use a common Chinese font, fallback to default if not found
        try:
            self.font = pygame.font.Font(font_path, 24)
        except IOError:
            self.font = pygame.font.SysFont("SimHei", 24) # Fallback for Chinese characters

    def draw_board(self):
        self.screen.fill(WHITE)

        # Draw grid lines
        for i in range(self.game.size + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen, BLACK,
                (self.padding, self.padding + i * self.cell_size),
                (self.width - self.padding, self.padding + i * self.cell_size),
                2
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, BLACK,
                (self.padding + i * self.cell_size, self.padding),
                (self.padding + i * self.cell_size, self.height - self.padding),
                2
            )

        # Pieces and controlled areas
        for i in range(self.game.size):
            for j in range(self.game.size):
                center = (
                    self.padding + j * self.cell_size + self.cell_size // 2,
                    self.padding + i * self.cell_size + self.cell_size // 2
                )

                if self.game.controlled[i, j] == 1:  # Player 1 (Red) controls
                    pygame.draw.rect(
                        self.screen, LIGHT_RED,
                        (self.padding + j * self.cell_size,
                         self.padding + i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )
                elif self.game.controlled[i, j] == -1:  # Player 2 (Blue) controls
                    pygame.draw.rect(
                        self.screen, LIGHT_BLUE,
                        (self.padding + j * self.cell_size,
                         self.padding + i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )

                # Draw pieces
                if self.game.board[i, j] == 1:  # Player 1 (Red) piece
                    pygame.draw.circle(self.screen, RED, center, self.radius)
                elif self.game.board[i, j] == -1:  # Player 2 (Blue) piece
                    pygame.draw.circle(self.screen, BLUE, center, self.radius)
                    #pygame.draw.circle(self.screen, BLACK, center, self.radius, 2) # Outline for Blue pieces

        # Round count
        round_text = f"回合: {self.game.round}/{self.game.max_rounds}"
        round_surface = self.font.render(round_text, True, BLACK)
        self.screen.blit(round_surface, (self.width - 150, 10))

        # Controlled areas display
        control_text = f"红方控制: {self.game.control_black}  蓝方控制: {self.game.control_white}" # Renamed for clarity
        control_surface = self.font.render(control_text, True, BLACK)
        self.screen.blit(control_surface, (10, 40))

        # Current player
        status_text = f"当前玩家: {'红方' if self.game.current_player == 1 else '蓝方'}"
        text_surface = self.font.render(status_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))

        # Game over status
        if self.game.game_over:
            if self.game.winner == 0:
                result_text = "游戏结束: 平局!"
            else:
                result_text = f"游戏结束: {'红方' if self.game.winner == 1 else '蓝方'}获胜!"
            result_surface = self.font.render(result_text, True, RED)
            text_rect = result_surface.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(result_surface, text_rect)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Convert mouse coordinates to board coordinates
                        x, y = event.pos
                        j = (x - self.padding) // self.cell_size
                        i = (y - self.padding) // self.cell_size

                        if 0 <= i < self.game.size and 0 <= j < self.game.size:
                            self.game.make_move((i, j))

            self.draw_board()
            pygame.display.flip()

        pygame.quit()
        sys.exit()