import pygame
from states import HealthState
import signal
import sys

class Visualizer:
    def __init__(self, model, radius=5, fps=30):
        pygame.init()

        self.model = model
        self.width = model.width * radius
        self.height = model.height * radius
        self.radius = radius
        
        self.header_height = self.height / 4

        self.window = pygame.display.set_mode((self.width, self.height + self.header_height))
        pygame.display.set_caption("Model Visualizer")

        self.clock = pygame.time.Clock()
        self.fps = fps

    def run(self):
        running = True
        
        def stop_simulation(_1 = None, _2 = None):
            nonlocal running
            running = False
            
        signal.signal(signal.SIGINT, stop_simulation)

        while running:
            self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_simulation()

            if hasattr(self.model, "step"):
                self.model.step()

            self.draw()

            pygame.display.flip()

        pygame.quit()
        
    def _draw_header(self):
        font = pygame.font.SysFont("Arial", 16)

        legend_items = [
            (HealthState.SUSCEPTIBLE, (0, 255, 0), "Susceptible"),
            (HealthState.EXPOSED, (255, 255, 0), "Exposed"),
            (HealthState.INFECTIOUS, (255, 0, 0), "Infectious"),
            (HealthState.RECOVERED, (0, 0, 0), "Recovered"),
        ]

        x_offset = self.radius
        y_offset = self.radius
        spacing_x = 160
        spacing_y = self.radius * 2
        x_position = 0
        y_position = 0
        
        texts = []

        for i, (state, color, label) in enumerate(legend_items):

            x = x_offset + x_position * spacing_x
            y = y_offset + y_position * spacing_y

            text = font.render(label, True, (0, 0, 0))
            text_w, text_h = text.get_size()
            if x + self.radius + text_w > self.width:
                x_position = 0
                y_position += 1
                x = x_offset
                y = y_offset + y_position * spacing_y

            x_position += 1
            
            texts.append((text, x + self.radius, y - 1, color))
            
        pygame.draw.rect(self.window, (255, 255, 255), (
            0, 0, self.width, self.header_height
        ))
        for text, x, y, color in texts:
            pygame.draw.rect(self.window, color, (x, y, self.radius, self.radius))
            self.window.blit(text, (x + self.radius, y))

    def draw(self):
        self.window.fill((255, 255, 255))
        self._draw_header()
        if hasattr(self.model, "agents"):
            for agent in self.model.agents:
                health_state = agent.health_state
                color = {
                    HealthState.SUSCEPTIBLE: (0, 255, 0),
                    HealthState.EXPOSED: (255, 255, 0),
                    HealthState.INFECTIOUS: (255, 0, 0),
                    HealthState.RECOVERED: (0, 0, 0)
                }[health_state]
                pygame.draw.circle(
                    self.window,
                    color,
                    (int(agent.x * self.radius + self.radius / 2), int(agent.y * self.radius + self.radius / 2) + self.header_height),
                    self.radius
                )