import numpy as np
import random
from collections import namedtuple, deque
# Importing the model
from dqn import DQN

import torch
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FrameStack:
    """Stack multiple frames to capture motion information (document line 1190-1195)"""
    def __init__(self, num_frames=4):
        self.frames = deque(maxlen=num_frames)
        self.num_frames = num_frames
    
    def reset(self, initial_frame):
        """Reset with initial frame, stack it num_frames times"""
        for _ in range(self.num_frames):
            self.frames.append(initial_frame)
    
    def add(self, frame):
        """Add new frame and return stacked frames"""
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)  # Shape: (4, 185, 95)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.cat([e.state for e in experiences if e is not None])
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(DEVICE)
        next_states = torch.cat(
            [e.next_state for e in experiences if e is not None])
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQNAgent():
    """Interacts with and learns form the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_every=100,
                 learning_starts=10000):  # SB3 uses 100k, we use 10k for 4GB GPU
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts  # Minimum steps before learning starts

        # Q- Network
        self.qnetwork_local = DQN(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = DQN(state_size, action_size, seed).to(DEVICE)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        
        # Target network update tracking (Hard update every C steps)
        self.hard_update_every = 10000  # DeepMind's choice for Atari (document line 1050)
        self.total_steps = 0  # Total training steps for target network update

    def preprocess_state(self, s):
        """
        preprocess gym images before storing them or passing them through the network.
        - from rgb to grayscale
        - normalize
        - crop
        - permute (h, w, c) to (c, h, w) as pytorch expects
        - to tensor
        """

        # transofrm image to grayscale
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3],
                          [0.2989, 0.5870, 0.1140])[..., np.newaxis] / 255

        state = rgb2gray(s.copy())
        # create tensor crop and permute image
        state = torch.from_numpy(state[15:200, 30:125, :].transpose(
            2, 0, 1)).float().unsqueeze(0).to(DEVICE)

        return state

    def step(self, state_, action, reward, next_state_, done):
        # ⚡ FRAME STACKING: state_ and next_state_ are already preprocessed tensors
        # No need to preprocess again
        if not isinstance(state_, torch.Tensor):
            # Old code path (raw numpy arrays)
            state = self.preprocess_state(state_.copy())
            next_state = self.preprocess_state(next_state_.copy())
        else:
            # New code path (already tensors from main.py)
            state = state_
            next_state = next_state_

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # ⚡ Increment total steps (for warm-up tracking)
        self.total_steps += 1

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # ⚡ SB3: Only start learning after learning_starts steps (warm-up period)
            if self.total_steps < self.learning_starts:
                return None  # Skip learning, just collect experiences
            
            # If enough samples are available in memory, get radom subset and learn
            if len(self.memory) > self.batch_size:
                experience = self.memory.sample()
                loss = self.learn(experience)
                
                # Clear GPU cache to prevent memory leak
                torch.cuda.empty_cache()
                
                return loss
        return None

    def act(self, state, eps=0, debug=False):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            debug (bool): if True, print Q-values
        """
        # ⚡ FRAME STACKING: State is already preprocessed 4-frame tensor from main.py
        # No need to preprocess again - just use it directly
        if not isinstance(state, torch.Tensor):
            # Old code path (if state is raw RGB numpy array)
            def rgb2gray(rgb):
                """transofrm rgb image to grayscale"""
                return np.dot(rgb[..., :3],
                              [0.2989, 0.5870, 0.1140])[..., np.newaxis]

            state = rgb2gray(state)
            # create tensor, crop, permute image's channels and send to DEVICE
            state = torch.from_numpy(state[15:200, 30:125, :].transpose(
                2, 0, 1)).float().unsqueeze(0).to(DEVICE)
        # else: state is already a tensor (4-frame stack from main.py)

        # freeze the q network to make predictions
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        if debug:
            print(f"    Q-values: {action_values.cpu().data.numpy()}")

        # Epsilon -greedy action selction
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            self.gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Clip rewards to prevent value explosion
        rewards = torch.clamp(rewards, -1.0, 1.0)

        # Use Huber loss for more stability (less sensitive to outliers than MSE)
        criterion = torch.nn.SmoothL1Loss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        # 🌟 DOUBLE DQN: Use local network to SELECT action, target network to EVALUATE
        with torch.no_grad():
            # Use local network to select best action for next state
            next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            # Use target network to evaluate Q-value of that action
            labels_next = self.qnetwork_target(next_states).gather(1, next_actions)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(DEVICE)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # Hard update every 10,000 steps (DeepMind's choice - document line 1050-1055)
        # Note: self.total_steps is incremented in step() method
        if self.total_steps % self.hard_update_every == 0:
            self.hard_update()
            print(f"🎯 Target network updated at step {self.total_steps}")

        # Get loss value and detach to prevent memory accumulation
        loss_value = loss.detach().item()
        
        # Delete tensors to free memory
        del states, actions, rewards, next_states, dones, predicted_targets, labels, loss
        
        return loss_value

        # def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # for target_param, local_param in zip(target_model.parameters(),
        #                                     local_model.parameters()):
        #    target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def hard_update(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
