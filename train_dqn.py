import argparse
import os
import random
import time
from collections import deque
from typing import Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import shipenv  # This registers the Ship-v0 environment


class DQN(nn.Module):
    """Deep Q-Network with fully connected layers."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay and target network."""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000, batch_size: int = 64,
                 target_update: int = 100, device: str = None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training stats
        self.step_count = 0
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and train if enough samples."""
        self.memory.push(state, action, reward, next_state, done)
        self.step_count += 1
        
        # Train every 4 steps
        if len(self.memory) > self.batch_size and self.step_count % 4 == 0:
            self.learn()
            
        # Update target network
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def learn(self):
        """Train the Q-network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_agent(env, agent, episodes: int, max_steps: int = 1000, 
                save_path: str = None):
    """Train the DQN agent."""
    
    scores = []
    scores_window = deque(maxlen=100)
    best_score = -np.inf
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Environment: {env.observation_space.shape[0]} state dims, {env.action_space.n} actions")
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        scores_window.append(score)
        
        # Decay epsilon once per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f'Episode {episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}')
            
            if avg_score > best_score:
                best_score = avg_score
                if save_path:
                    torch.save(agent.q_network.state_dict(), save_path)
                    print(f'New best model saved! Score: {best_score:.2f}')
    
    return scores


def test_agent(env, agent, episodes: int = 5, render: bool = True):
    """Test the trained agent."""
    
    print(f"\nTesting agent for {episodes} episodes...")
    
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        step = 0
        
        while True:
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for visualization
            
            action = agent.act(state, training=False)  # No exploration during testing
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            step += 1
            
            if done:
                break
        
        scores.append(score)
        print(f'Episode {episode + 1}: Score = {score:.2f}, Steps = {step}')
        
        if render:
            time.sleep(1)  # Pause between episodes
    
    print(f'Average test score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    return scores


def plot_training(scores: List[float], save_path: str = None):
    """Plot training progress."""
    plt.figure(figsize=(10, 6))
    
    # Plot raw scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = min(100, len(scores) // 10)
    if window_size > 1:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg)
        plt.title(f'Moving Average (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train DQN on Ship Environment')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--target_update', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--save_model', type=str, default='dqn_model.pth', help='Path to save model')
    parser.add_argument('--plot_path', type=str, default='training_progress.png', help='Path to save training plot')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training (slower, for visualization)')
    parser.add_argument('--no_test_render', action='store_true', help='Disable rendering during testing')
    
    args = parser.parse_args()
    
    # Create environment - use None for fast training by default, "human" only if explicitly requested
    env = gym.make("Ship-v0", render_mode="human" if args.render else None)
    
    # Get environment dimensions dynamically
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment loaded:")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    # Get the unwrapped environment to access parameters
    unwrapped_env = env.unwrapped
    print(f"  dt: {unwrapped_env.P.dt:.3f} seconds")
    print(f"  max_steps: {unwrapped_env.P.max_steps}")
    if unwrapped_env.render_mode is None:
        print(f"  Mode: FAST (no rendering overhead)")
    else:
        print(f"  Mode: SLOW (with rendering)")
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )
    
    # Train agent
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    scores = train_agent(
        env=env,
        agent=agent,
        episodes=args.episodes,
        save_path=args.save_model
    )
    
    # Plot training progress
    plot_training(scores, args.plot_path)
    
    # Test agent
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    # Load best model if it exists
    if os.path.exists(args.save_model):
        agent.q_network.load_state_dict(torch.load(args.save_model))
        print(f"Loaded best model from {args.save_model}")
    
    # For testing, create environment with rendering unless explicitly disabled
    test_render_mode = None if args.no_test_render else "human"
    test_env = gym.make("Ship-v0", render_mode=test_render_mode)
    
    test_scores = test_agent(
        env=test_env,
        agent=agent,
        episodes=args.test_episodes,
        render=not args.no_test_render
    )
    
    env.close()
    test_env.close()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Final average test score: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    print(f"Model saved to: {args.save_model}")
    print(f"Training plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
