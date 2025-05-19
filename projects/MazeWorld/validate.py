import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
import sys
from src.models import E2EObjNav
from src.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MazeEpochCausal

if __name__ == "__main__":
    runner=Runner()
    runner.start(E2EObjNav, [], [MazeEpochCausal], 'validate')
