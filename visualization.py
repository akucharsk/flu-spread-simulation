import pygame

class Visualizer:
    def __init__(self, model, width=800, height=600, fps=30):
        pygame.init()

        self.model = model
        self.width = width
        self.height = height

        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Model Visualizer")

        self.clock = pygame.time.Clock()
        self.fps = fps

    def run(self):
        running = True

        while running:
            self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if hasattr(self.model, "step"):
                self.model.step()

            self.draw()

            pygame.display.flip()

        pygame.quit()

    def draw(self):
        self.window.fill((255, 255, 255))

        if hasattr(self.model, "agents"):
            for agent in self.model.agents:
                pygame.draw.circle(
                    self.window,
                    getattr(agent, "color", (0, 0, 255)),
                    (int(agent.x), int(agent.y)),
                    getattr(agent, "radius", 5),
                )