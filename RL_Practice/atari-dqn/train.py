
import copy
import random
from pathlib import Path

import gymnasium as gym
import ale_py
import numpy as np
import torch

from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation
)

from src.dqn_network import DQNetwork
from src.replay_buffer import ReplayBuffer
from src.train_step import train_step


# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42

TOTAL_STEPS = 100_000

REPLAY_CAPACITY = 1000
LEARNING_STARTS = 500
BATCH_SIZE = 32

GAMMA = 0.99

LEARNING_RATE = 1e-4

EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999995

TARGET_UPDATE_FREQUENCY = 1000

CHECKPOINT_DIR = Path("checkpoints")


# ==========================================
# 2. REPRODUCIBILITY AND DEVICE
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 3. CREATE ATARI ENVIRONMENT
# ==========================================

gym.register_envs(ale_py)

env = gym.make(
    "ALE/Pong-v5",
    frameskip=1
)

env = AtariPreprocessing(
    env,
    frame_skip=4,
    screen_size=84,
    grayscale_obs=True,
    scale_obs=False
)

env = FrameStackObservation(
    env,
    stack_size=4
)

env.action_space.seed(SEED)

state, info = env.reset(seed=SEED)

num_actions = env.action_space.n

print("Observation shape:", state.shape)
print("Number of actions:", num_actions)


# ==========================================
# 4. CREATE ONLINE AND TARGET NETWORKS
# ==========================================

online_network = DQNetwork(
    input_channels=4,
    num_actions=num_actions
).to(device)

target_network = copy.deepcopy(
    online_network
).to(device)

# Target network is not optimized directly
target_network.eval()

for parameter in target_network.parameters():
    parameter.requires_grad = False


# ==========================================
# 5. CREATE OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    online_network.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. CREATE REPLAY BUFFER
# ==========================================

replay_buffer = ReplayBuffer(
    capacity=REPLAY_CAPACITY
)


# ==========================================
# 7. EPSILON-GREEDY ACTION SELECTION
# ==========================================

def select_action(
    state,
    online_network,
    epsilon,
    num_actions,
    device
):

    # Exploration: choose a random action
    if random.random() < epsilon:

        return random.randrange(num_actions)

    # Convert observation to a PyTorch tensor
    state_array = np.asarray(
        state,
        dtype=np.uint8
    )

    state_tensor = torch.from_numpy(
        state_array.copy()
    ).unsqueeze(0)

    # Normalize pixel values
    state_tensor = (
        state_tensor.float().to(device) / 255.0
    )

    # Exploitation: choose the highest Q-value
    with torch.no_grad():

        q_values = online_network(state_tensor)

    return q_values.argmax(dim=1).item()


# ==========================================
# 8. TRAINING VARIABLES
# ==========================================

training_steps = 0
episode_count = 0

episode_reward = 0
episode_length = 0

episode_rewards = []

last_loss = None


# ==========================================
# 9. CHECKPOINT SAVING
# ==========================================

def save_checkpoint(
    path,
    online_network,
    target_network,
    optimizer,
    training_steps,
    epsilon
):

    torch.save(
        {
            "online_network": online_network.state_dict(),
            "target_network": target_network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_steps": training_steps,
            "epsilon": epsilon
        },
        path
    )

    print("Checkpoint saved:", path)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 10. MAIN TRAINING LOOP
# ==========================================

try:

    for step in range(1, TOTAL_STEPS + 1):

        # ----------------------------------
        # A. Select action
        # ----------------------------------

        action = select_action(
            state,
            online_network,
            EPSILON,
            num_actions,
            device
        )

        # ----------------------------------
        # B. Interact with environment
        # ----------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        # ----------------------------------
        # C. Store experience
        # ----------------------------------

        replay_buffer.push(
            np.asarray(state, dtype=np.uint8).copy(),
            action,
            reward,
            np.asarray(next_state, dtype=np.uint8).copy(),
            terminated
        )

        # ----------------------------------
        # D. Update episode statistics
        # ----------------------------------

        episode_reward += reward
        episode_length += 1

        # ----------------------------------
        # E. Train the online network
        # ----------------------------------

        if replay_buffer.len() >= LEARNING_STARTS:

            batch = replay_buffer.sample(
                BATCH_SIZE
            )

            last_loss = train_step(
                batch,
                online_network,
                target_network,
                optimizer,
                GAMMA,
                device
            )

            training_steps += 1

            # ----------------------------------
            # F. Synchronize target network
            # ----------------------------------

            if training_steps % TARGET_UPDATE_FREQUENCY == 0:

                target_network.load_state_dict(
                    online_network.state_dict()
                )

                print(
                    f"Target network updated at "
                    f"training step {training_steps}"
                )

        # ----------------------------------
        # G. Decay exploration
        # ----------------------------------

        EPSILON = max(
            EPSILON_MIN,
            EPSILON * EPSILON_DECAY
        )

        # ----------------------------------
        # H. Move to next state
        # ----------------------------------

        state = next_state

        # ----------------------------------
        # I. Handle episode ending
        # ----------------------------------

        if terminated or truncated:

            episode_count += 1

            episode_rewards.append(
                episode_reward
            )

            average_reward = np.mean(
                episode_rewards[-10:]
            )

            print(
                f"Episode: {episode_count} | "
                f"Environment step: {step} | "
                f"Length: {episode_length} | "
                f"Reward: {episode_reward:.2f} | "
                f"Average reward (last 10): {average_reward:.2f} | "
                f"Epsilon: {EPSILON:.4f} | "
                f"Loss: {last_loss}"
            )

            # Save periodically
            if episode_count % 10 == 0:

                save_checkpoint(
                    CHECKPOINT_DIR / "latest.pt",
                    online_network,
                    target_network,
                    optimizer,
                    training_steps,
                    EPSILON
                )

            # Reset episode statistics
            episode_reward = 0
            episode_length = 0

            # Reset environment
            state, info = env.reset()

finally:

    # Save final model
    save_checkpoint(
        CHECKPOINT_DIR / "final.pt",
        online_network,
        target_network,
        optimizer,
        training_steps,
        EPSILON
    )

    env.close()

    print("Training finished.")
