"""
Entry points have been split into dedicated scripts:

  python main_simulation.py --steps N   Run headless simulation for N steps
  python main_visualization.py           Run with interactive Solara visualization
"""
import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(1)
