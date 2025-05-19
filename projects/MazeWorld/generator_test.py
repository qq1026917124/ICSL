import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import sys
from src.models import E2EObjNav
from src.utils import GeneratorRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MAZEGenerator, general_generator

if __name__ == "__main__":
    print("Start generator test")
    print(f"Visible GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    runner=GeneratorRunner()
    runner.start(E2EObjNav, general_generator)