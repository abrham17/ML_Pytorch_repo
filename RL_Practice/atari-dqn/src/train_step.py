import torch
import torch.nn.functional as F

def train_step(
    batch,
    online_network,
    target_network,
    optimizer,
    gamma,
    device
):

    # Separate the experiences
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.from_numpy(
        __import__("numpy").stack(states)
    ).to(device)

    next_states = torch.from_numpy(
        __import__("numpy").stack(next_states)
    ).to(device)

    actions = torch.tensor(
        actions,
        dtype=torch.long,
        device=device
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device
    )

    # Normalize Atari images
    states = states.float() / 255.0
    next_states = next_states.float() / 255.0

    # --------------------------------
    # 1. Online network prediction
    # --------------------------------

    q_values = online_network(states)

    current_q_values = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # --------------------------------
    # 2. Target network prediction
    # --------------------------------

    with torch.no_grad():

        next_q_values = target_network(
            next_states
        )

        max_next_q_values = next_q_values.max(
            dim=1
        ).values

        # --------------------------------
        # 3. Bellman target
        # --------------------------------

        targets = (
            rewards
            + gamma
            * (1 - dones)
            * max_next_q_values
        )

    # --------------------------------
    # 4. Loss
    # --------------------------------

    loss = F.smooth_l1_loss(
        current_q_values,
        targets
    )

    # --------------------------------
    # 5. Update online network
    # --------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()