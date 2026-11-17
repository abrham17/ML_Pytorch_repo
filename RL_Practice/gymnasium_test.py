import gymnasium as gym
import matplotlib.pyplot as plt

# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Reset environment
observation, info = env.reset()

print("Initial observation:")
print(observation)

total_reward = 0

for i in range(100):

    # Choose a random action
    action = env.action_space.sample()

    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Add reward
    total_reward += reward

    print(f"Total reward: {total_reward}")

    # Check whether the episode has ended
    if terminated or truncated:
        observation, info = env.reset()

# Get the rendered frame
frame = env.render()

# Display it
plt.imshow(frame)
plt.axis("off")
plt.show()

env.close()