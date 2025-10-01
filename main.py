import gymnasium as gym
import gymnasium as gym
import shipenv  # ensure this triggers your register("Ship-v0", ...)

env = gym.make("Ship-v0", render_mode="human")
obs, info = env.reset()

done = False
while not done:
    #action = env.action_space.sample()
    action = 7
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
