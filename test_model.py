#!/usr/bin/env python3
"""
Test script to load the most recently saved DQN model and test it on a random episode.
"""

import argparse
import os
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import shipenv  # This registers the Ship-v0 environment


class DQN(nn.Module):
    """Deep Q-Network with fully connected layers."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: list = [64, 32, 16]):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer: state_size -> hidden_size[0]
        self.layers.append(nn.Linear(state_size, hidden_size[0]))
        
        # Hidden layers: hidden_size[i] -> hidden_size[i+1]
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        
        # Output layer: hidden_size[-1] -> action_size
        self.layers.append(nn.Linear(hidden_size[-1], action_size))
        
    def forward(self, x):
        # Apply ReLU to all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        # Final layer without ReLU
        return self.layers[-1](x)


class TestAgent:
    """Simple agent for testing loaded models."""
    
    def __init__(self, state_size: int, action_size: int, model_path: str, device: str = None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        self.q_network = DQN(state_size, action_size).to(self.device)
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.q_network.eval()  # Set to evaluation mode
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using the trained Q-network (no exploration)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()


def find_most_recent_model(model_dir: str = ".", model_prefix: str = "dqn_model") -> str:
    """Find the most recently saved model file."""
    model_files = []
    
    # Look for .pth files with the model prefix
    for file in os.listdir(model_dir):
        if file.startswith(model_prefix) and file.endswith('.pth'):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                model_files.append((file_path, os.path.getmtime(file_path)))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found with prefix '{model_prefix}' in {model_dir}")
    
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x[1], reverse=True)
    most_recent = model_files[0][0]
    
    print(f"Found {len(model_files)} model file(s):")
    for file_path, mtime in model_files:
        print(f"  {file_path} (modified: {time.ctime(mtime)})")
    print(f"Using most recent: {most_recent}")
    
    return most_recent


def test_episode(env, agent, render: bool = True, max_steps: int = 1000, verbose: bool = False) -> Tuple[float, int, bool, bool]:
    """Run a single test episode and return score, steps, success, and crash status."""
    state, info = env.reset()
    score = 0
    steps = 0
    
    print(f"Starting episode - Goal: {info['goal']}, Rocks: {len(info['rocks'])}")
    if verbose:
        print("State format: [x, y, dx, dy, v] + ray_distances (if enabled)")
        print("Actions: 0=Turn Left, 1=Turn Right, 2=Accelerate, 3=Brake")
    
    while steps < max_steps:
        if render:
            env.render()
            time.sleep(0.05)  # Slow down for visualization
        
        # Get action from trained agent
        action = agent.act(state)
        
        # Print current state if verbose
        if verbose:
            print(f"  Step {steps+1}: State={state}, Action={action}")
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = next_state
        score += reward
        steps += 1
        
        if done:
            success = info.get('reached_goal', False)
            crashed = info.get('destroyed', False)
            print(f"Episode finished after {steps} steps")
            print(f"  Final score: {score:.2f}")
            print(f"  Success: {success}")
            print(f"  Crashed: {crashed}")
            print(f"  Final distance to goal: {info.get('distance', 'N/A'):.2f}")
            return score, steps, success, crashed
    
    print(f"Episode truncated after {max_steps} steps")
    return score, steps, False, False


def main():
    parser = argparse.ArgumentParser(description='Test trained DQN model on Ship Environment')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to model file (if None, finds most recent)')
    parser.add_argument('--episodes', type=int, default=1, 
                       help='Number of test episodes to run')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Enable rendering during testing')
    parser.add_argument('--no_render', action='store_true', 
                       help='Disable rendering (overrides --render)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--model_dir', type=str, default='.',
                       help='Directory to search for model files')
    parser.add_argument('--model_prefix', type=str, default='dqn_model',
                       help='Prefix for model files to search for')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path is None:
        try:
            model_path = find_most_recent_model(args.model_dir, args.model_prefix)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
    
    # Create environment
    render_mode = "human" if args.render and not args.no_render else None
    env = gym.make("Ship-v0", render_mode=render_mode)
    
    # Get environment dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment loaded:")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Render mode: {render_mode}")
    
    # Create and load agent
    try:
        agent = TestAgent(state_size, action_size, model_path)
    except Exception as e:
        print(f"Error loading agent: {e}")
        env.close()
        return
    
    # Run test episodes
    print(f"\n{'='*50}")
    print(f"TESTING PHASE - {args.episodes} episode(s)")
    print(f"{'='*50}")
    
    scores = []
    steps_list = []
    successes = []
    crashes = []
    
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        print("-" * 30)
        
        score, steps, success, crashed = test_episode(
            env, agent, 
            render=args.render and not args.no_render,
            max_steps=args.max_steps,
            verbose=True
        )
        
        scores.append(score)
        steps_list.append(steps)
        successes.append(success)
        crashes.append(crashed)
        
        if args.render and not args.no_render:
            time.sleep(1)  # Pause between episodes
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Episodes: {args.episodes}")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Success Rate: {np.mean(successes):.1%} ({sum(successes)}/{args.episodes})")
    print(f"Crash Rate: {np.mean(crashes):.1%} ({sum(crashes)}/{args.episodes})")
    
    if args.episodes > 1:
        print(f"\nIndividual episode results:")
        for i, (score, steps, success, crashed) in enumerate(zip(scores, steps_list, successes, crashes)):
            status = "SUCCESS" if success else "CRASHED" if crashed else "TIMEOUT"
            print(f"  Episode {i+1}: Score={score:.2f}, Steps={steps}, Status={status}")
    
    env.close()
    print(f"\nModel tested successfully using: {model_path}")


if __name__ == "__main__":
    main()
