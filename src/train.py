import os
import itertools
from src.env import make_env
from src.agent import DQNAgent
from src.buffer import ReplayBuffer

def train():
    env = make_env("ALE/Pong-v5")
    
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape

    agent = DQNAgent(input_shape, n_actions, lr=1e-4, gamma=0.99, batch_size=32)
    buffer = ReplayBuffer(capacity=100_000)

    # 超参数
    epsilon_start = 1.0
    epsilon_final = 0.02
    epsilon_decay_frames = 100_000
    target_update_freq = 1000
    max_frames = 1_000_000
    
    epsilon = epsilon_start
    episode_reward = 0
    episode_idx = 0
    
    state, info = env.reset()
    
    print("Starting Training...")
    
    for frame_idx in range(1, max_frames + 1):
        
        epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay_frames)
        
        
        action = agent.get_action(state, epsilon)
        next_state, reward, target, truncated, info = env.step(action)
        done = target or truncated
        
        
        buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        
        if frame_idx > 10000:
            loss = agent.learn(buffer)
        
        if frame_idx % target_update_freq == 0:
            agent.update_target_network()
            
        if done:
            print(f"Frame {frame_idx} | Episode {episode_idx} | Reward: {episode_reward} | Epsilon: {epsilon:.3f}")
            state, info = env.reset()
            episode_reward = 0
            episode_idx += 1
            
            if episode_idx % 20 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                agent.save(f"checkpoints/dqn_pong_{episode_idx}.pt")

if __name__ == "__main__":
    train()
