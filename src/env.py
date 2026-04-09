import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
def make_env(env_id="ALE/Pong-v5", render_mode=None):

    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True,
    )
    
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    return env
