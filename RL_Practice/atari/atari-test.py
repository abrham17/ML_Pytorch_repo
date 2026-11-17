
import gymnasium as gym
import ale_py
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. Register Atari environments
gym.register_envs(ale_py)

# 2. Create the Pong environment
env = gym.make("ALE/Pong-v5")

# 3. Reset environment
observation, info = env.reset()

# 4. Convert RGB image to grayscale
gray_image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

# 5. Normalize pixel values (0-255 -> 0-1)
normalized_observation = gray_image.astype(np.float32) / 255.0

# 6. Convert NumPy array to PyTorch tensor
image_tensor = torch.from_numpy(normalized_observation)

# 7. Add channel dimension (1, Height, Width)
image_tensor = image_tensor.unsqueeze(0)

# 8. Display image information
print("Environment created successfully!")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Action meanings:", env.unwrapped.get_action_meanings())

print("\nOriginal observation shape:", observation.shape)
print("Grayscale image shape:", gray_image.shape)
print("Tensor shape:", image_tensor.shape)
print("Tensor dtype:", image_tensor.dtype)
print("Normalized pixel range:",
      normalized_observation.min(),
      normalized_observation.max())

resized = cv2.resize(
    gray_image,
    (84, 84),
    interpolation=cv2.INTER_AREA
)

print(resized.shape)
# 9. Visualize original and processed images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(observation)
plt.title("Original Pong Observation (RGB)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_tensor.squeeze(0).numpy(), cmap="gray")
plt.title("Processed Observation (Grayscale)")
plt.axis("off")

plt.tight_layout()
plt.show()

# 10. Close environment
env.close()
