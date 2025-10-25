"""
Script để xem AI chơi Space Invaders LIVE với render mode
"""
import gymnasium as gym
import ale_py
import torch
import numpy as np
from agent import DQNAgent, FrameStack, DEVICE

# Register ALE environments
gym.register_envs(ale_py)

def play_live(model_path='./train/model_ep9050.pth', n_games=10):
    """
    Xem AI chơi game live với visualization
    
    Args:
        model_path: Path to trained model
        n_games: Number of games to play
    """
    # Create environment with human render mode
    env = gym.make('SpaceInvaders-v4', render_mode='human')
    
    # Create agent
    agent = DQNAgent(state_size=4,
                     action_size=env.action_space.n,
                     seed=0)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent.qnetwork_local.load_state_dict(torch.load(model_path))
    agent.qnetwork_local.eval()
    print("✅ Model loaded!")
    
    # Play games
    for game in range(n_games):
        print(f"\n{'='*50}")
        print(f"GAME {game + 1}/{n_games}")
        print(f"{'='*50}")
        
        # ⚡ Initialize frame stack
        frame_stack = FrameStack(num_frames=4)
        raw_observation, _ = env.reset()
        
        # Process initial frame
        processed_frame = agent.preprocess_state(raw_observation).cpu().numpy().squeeze()
        frame_stack.reset(processed_frame)
        observation = torch.from_numpy(np.stack(frame_stack.frames, axis=0)).float().unsqueeze(0).to(DEVICE)
        
        score = 0
        done = False
        step = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Track actions
        
        while not done:
            # Get action from agent (no exploration, greedy policy)
            action = agent.act(observation, eps=0.0)
            action_counts[action] += 1
            
            # Take action
            raw_observation, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # ⚡ Process next frame with stacking
            processed_frame = agent.preprocess_state(raw_observation).cpu().numpy().squeeze()
            frame_stack.add(processed_frame)
            observation = torch.from_numpy(np.stack(frame_stack.frames, axis=0)).float().unsqueeze(0).to(DEVICE)
            done = done or truncated
            
            score += reward
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step}: Score = {score:.0f}")
        
        # Action names for Space Invaders
        action_names = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT', 4: 'RIGHTFIRE', 5: 'LEFTFIRE'}
        
        print(f"\n🎮 GAME {game + 1} OVER!")
        print(f"   Final Score: {score:.0f}")
        print(f"   Total Steps: {step}")
        print(f"   📊 Action Distribution:")
        for action_id, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            percentage = (count / step) * 100
            print(f"      {action_names[action_id]:10s}: {count:4d} ({percentage:5.1f}%)")
    
    env.close()
    print("\n✅ All games completed!")


if __name__ == '__main__':
    # Configuration
    import sys
    MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else './train/model_ep9500.pth'
    N_GAMES = 5
    
    print("🎮 DQN Agent - Space Invaders Live Play (REWARD SHAPING)")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Games: {N_GAMES}")
    print("=" * 50)
    print("\nPress Ctrl+C to stop early")
    print("\n")
    
    try:
        play_live(model_path=MODEL_PATH, n_games=N_GAMES)
    except KeyboardInterrupt:
        print("\n\n⚠️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
