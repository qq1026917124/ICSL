import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from src.models import E2EObjNav
from src.utils import Runner
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MazeEpochVAE, MazeEpochCausal

if __name__ == "__main__":
    torch.cuda.empty_cache()
    runner=Runner()
    print(f"Visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    runner.start(E2EObjNav, [MazeEpochCausal], [MazeEpochCausal])
