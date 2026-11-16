import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()

print("Initial observation:")
print(observation)
total_reward = 0

for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Total reward: {total_reward}")
    if terminated or truncated:
        observation, info = env.reset()


env.close()