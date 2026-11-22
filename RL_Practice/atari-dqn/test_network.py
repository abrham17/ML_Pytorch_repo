
import copy
import torch

from dqn_network import DQNetwork


# Use CPU for this implementation
device = torch.device("cpu")

# Atari input: 4 stacked frames
input_channels = 4

# Example: Pong has a discrete action space
num_actions = 6


# Create online network
online_network = DQNetwork(
    input_channels,
    num_actions
).to(device)


# Create target network as a copy
target_network = copy.deepcopy(online_network).to(device)


# Target network does not need gradients
target_network.eval()

for parameter in target_network.parameters():
    parameter.requires_grad = False


# Test with a fake batch of 1 observation
dummy_state = torch.zeros(
    1, 4, 84, 84,
    dtype=torch.float32
).to(device)


# Get Q-values from online network
q_values = online_network(dummy_state)


print("Input shape:", dummy_state.shape)
print("Output shape:", q_values.shape)
print("Q-values:", q_values)


# Check the two networks have the same initial weights
same_weights = all(
    torch.equal(p1, p2)
    for p1, p2 in zip(
        online_network.parameters(),
        target_network.parameters()
    )
)

print("Initial weights identical:", same_weights)
