from model import EpidemicModel
from visualization import Visualizer

model = EpidemicModel(population=1000, width=40, height=40)

# for i in range(1000):
#     model.step()

#     infected = sum(
#         1 for a in model.schedule.agents
#         if a.health_state.value == "I"
#     )

#     print(f"Step {i}: {infected} infected")
visualizer = Visualizer(model, fps=10)
visualizer.run()
