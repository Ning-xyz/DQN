import torch
import time
from src.env import make_env
from src.agent import DQNAgent

def evaluate(model_path):

    env = make_env("ALE/Pong-v5", render_mode="human")
    
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape
    
    agent = DQNAgent(input_shape, n_actions)
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return
        
    state, info = env.reset()
    done = False
    episode_reward = 0
    
    print("Starting evaluation episode...")
    while not done:
        action = agent.get_action(state, epsilon=0.0)
        state, reward, target, truncated, info = env.step(action)
        done = target or truncated
        episode_reward += reward
        time.sleep(0.02)
        
    print(f"Evaluation completed. Total Reward: {episode_reward}")
    env.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        print("Usage: python -m src.eval <path_to_checkpoint>")
