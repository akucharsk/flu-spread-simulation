import solara
import networkx as nx
import matplotlib.pyplot as plt


@solara.component
def NetworkView(model):
    G = model.grid.G

    if not hasattr(model, "pos"):
        model.pos = nx.spring_layout(G, seed=42)

    pos = model.pos

    fig, ax = plt.subplots(figsize=(5, 5))

    node_colors = []
    for n in G.nodes:
        agents = model.grid.get_cell_list_contents([n])

        if any(a.health_state.name == "INFECTIOUS" for a in agents):
            node_colors.append("red")
        else:
            node_colors.append("skyblue")

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=800,
        edge_color="gray",
        ax=ax,
    )

    solara.FigureMatplotlib(fig)