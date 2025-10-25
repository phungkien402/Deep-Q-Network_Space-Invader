# DQN for Space Invaders - Thesis Project# dqn-on-space-invaders



Deep Q-Network implementation for playing Atari Space Invaders using PyTorch.## Overview



## 📋 Project StructureThis is a PyTorch implementation of a Deep Q-Network agent trained to play the Atari 2600 game of Space Invaders. The related paper is the following: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf), published in 2014 by Google Deepmind. 



```This repository also corresponds to the source code for this [post](https://qarchli.github.io/2020-06-04-dqn-to-play-space-invaders/) I have written on the subject.

dqn-on-space-invaders-master/

├── main.py                 # Main training script with Double DQN## Dependencies

├── agent.py                # DQN Agent implementation with experience replay

├── dqn.py                  # Neural network architecture (CNN)Install the requirements using this command:

├── play_live.py            # Test trained models and visualize gameplay

├── plot_checkpoint.py      # Visualize training progress from checkpoints```bash

├── requirements.txt        # Python dependenciespip install -r requirements.txt

├── train/                  # Training checkpoints and saved models```

│   ├── model_ep*.pth      # Model weights at different episodes

│   └── training_state_ep*.pkl  # Complete training state snapshotsThere is one more thing to install to have access to the Atari environment. In fact, OpenAI gym library does not support by default the Atari environment. 

└── README.md              # This file

```### Linux users



## 🚀 Key FeaturesSimply run the following command:

```bash

### 1. **Double DQN Architecture**pip install atari-py

- **Local Network**: Selects best actions```

- **Target Network**: Evaluates Q-values (updated every 10,000 steps)### Windows users

- Reduces overestimation bias compared to vanilla DQN

Start by running the same command as Linux users, if you have some errors popping up then detailed instructions to install Atari environments in Windows platforms are given [here](https://github.com/Kojoley/atari-py).    

### 2. **Frame Stacking**

- Stacks 4 consecutive frames (185×95×4)## Usage

- Provides temporal information for motion perception

- Essential for understanding game dynamics from static imagesOnce dependencies are installed, you can open `main.py` and decide whether you want to train or test the agent. This can be done by setting the `TRAIN` variable to either `True`or `False`. Other hyper-parameters are to be specified in the same file.



### 3. **Frame Skipping**If trained, the agent's weights are saved in `./train`. Otherwise, videos of the agent playing are stored in `./test/`.

- Repeats each action for 4 frames (DeepMind standard)

- Reduces computational cost by 4x## Results

- Matches human reaction time

Below are the curves of the scores obtained throughout the training phase by the DQN agent as well as a random agent used as a baseline:

### 4. **Reward Shaping**

Multi-objective reward design to guide learning:<p align='center'>

- **Life Loss Penalty (-5.0)**: Strong survival incentive  <img src="./assets/scores.jpg"/>

- **Survival Bonus (+0.01)**: Encourage longer episodes</p>

- **Shooting Bonus (+0.05)**: Promote offensive play

- **Movement Bonus (+0.03)**: Encourage tactical positioningThe DQN agent has played 100 episodes, 10000 timesteps each, and it has been able to improve its decision-making process as the training progresses. In fact, it starts by randomly selecting actions, waiting for the replay buffer to be sufficiently full to start the training. After several episodes of playing, the agent starts showing learning improvements and rather satisfactory results by the end of the training. This is due to the fact that its policy becomes progressively less random, as the update rule encourages it to exploit actions with higher rewards. 

- **Inaction Penalty (-0.01)**: Discourage passivity

Here is a game where the agent is playing after being trained: 

### 5. **Experience Replay**

- Buffer size: 20,000 transitions (optimized for 4GB GPU)

- Batch size: 24<p align='center'>

- Breaks temporal correlations in training data  <img src="./assets/game.gif"/>

</p>

### 6. **Epsilon-Greedy Exploration**

- Start: ε = 1.0 (100% exploration)It has done a pretty good job overall. Nevertheless, it has to be trained more and perhaps get its policy network tuned so that it can get a higher score.

- End: ε = 0.01 (1% exploration)

- Decay: 0.995 per episode (~5% after 500 episodes)## TODO

- [ ] Add the possibility of hyper-parameters tuning.

## 🛠️ Installation- [ ] TensorBoard support.

- [ ] Add a run manager.

```bash

# Install dependencies## Resources

pip install -r requirements.txt

```[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf )



## 📊 Training[Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)



### Start New Training[How RL agents learn to play Atari games](https://www.youtube.com/watch?v=rbsqaJwpu6A&feature=youtu.be&t=9m55s)

```bash

python main.py[Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
```

### Continue from Checkpoint
Edit `main.py`:
```python
LOAD_MODEL = True  # Will auto-load latest checkpoint
```

### Interactive Epsilon Adjustment
When continuing training:
```
🔍 Change epsilon? (y/n): y
Enter new epsilon (current: 0.01): 0.1
```

### Keyboard Shortcuts
- **Ctrl+C**: Stop training and show progress graph

## 📈 Monitoring & Analysis

### Visualize Training Progress
```bash
python plot_checkpoint.py
```

### Test Trained Model
```bash
python play_live.py ./train/model_ep2000.pth 3
```

## 📁 Checkpoint Contents

Each checkpoint contains complete training state:
- Performance metrics (scores, losses, avg_score)
- Hyperparameters (buffer_size, learning_rate, etc.)
- Training progress (episode, epsilon, total_steps)
- Metadata (timestamp, random_seed)

## 🎮 Game Information

**Actions:** NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE  
**Observation:** 185×95×4 stacked grayscale frames  
**Rewards:** +10-30 (enemies), +50-300 (UFO)

## 🧠 Neural Network

```
Input: 185×95×4 → Conv(32) → Conv(64) → Conv(64) → FC(512) → FC(6)
```

## 📊 Expected Results

- **Average Score**: ~400 (after 2000+ episodes)
- **Max Score**: 1000+
- **Training Time**: ~50 hours for 10,000 episodes (GTX 1050 Ti)

## 👤 Author

Thesis Project - October 2025
