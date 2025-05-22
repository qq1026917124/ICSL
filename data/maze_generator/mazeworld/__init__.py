from gym.envs.registration import register

from .envs import MazeWorldContinuous3D
from .envs import MazeTaskSampler, Resampler, MazeStaticSampler, MazeSampleFromCellWalls

register(
    id='mazeworld-v2',
    entry_point=MazeWorldContinuous3D,
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (256, 256),
        "max_steps": 5000,
        "visibility_3D": 12.0,
        "command_in_observation": False,
        "action_space_type": "Discrete16" ,
    }
)
