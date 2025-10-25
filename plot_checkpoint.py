"""
Script to plot training progress from checkpoint files
Usage: python plot_checkpoint.py <checkpoint_path>
Example: python plot_checkpoint.py train/training_state_ep1000.pkl
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import glob
import os

def plot_from_checkpoint(checkpoint_path):
    """Load training state from checkpoint and plot graphs"""
    
    # Load training state
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        training_state = pickle.load(f)
    
    scores = training_state['scores']
    done_timesteps = training_state['done_timesteps']
    episode = training_state['episode']
    epsilon = training_state['epsilon']
    
    print(f"✅ Loaded {len(scores)} episodes (up to episode {episode})")
    print(f"   Final epsilon: {epsilon:.4f}")
    print(f"   Latest score: {scores[-1]:.2f}")
    print(f"   Average score (last 100): {np.mean(scores[-100:]):.2f}")
    
    # Calculate cumulative timesteps
    cumulative_timesteps = np.cumsum(done_timesteps)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ===== PLOT 1: Scores vs Episodes =====
    N = 100  # running mean window
    if len(scores) >= N:
        smoothed_scores = np.convolve(np.array(scores), np.ones((N,)) / N, mode='valid')
        episode_indices = np.arange(N-1, len(scores))
        
        ax1.plot(episode_indices, smoothed_scores, linewidth=2, color='blue', label=f'{N}-Episode Running Mean')
        ax1.scatter(range(len(scores)), scores, alpha=0.1, s=5, color='lightblue', label='Raw Scores')
        ax1.set_xlabel('Episode #', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title(f'DQN Training Progress (Episode {episode}) - Score vs Episodes', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        max_score = max(scores)
        max_episode = scores.index(max_score)
        avg_last_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        stats_text = f'Max Score: {max_score:.0f} (Ep {max_episode})\nAvg (last 100): {avg_last_100:.2f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== PLOT 2: Scores vs Timesteps =====
    if len(scores) >= N:
        smoothed_timesteps = np.convolve(cumulative_timesteps, np.ones((N,)) / N, mode='valid')
        
        ax2.plot(smoothed_timesteps, smoothed_scores, linewidth=2, color='green', label=f'{N}-Episode Running Mean')
        ax2.scatter(cumulative_timesteps, scores, alpha=0.1, s=5, color='lightgreen', label='Raw Scores')
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'DQN Training Progress - Score vs Timesteps', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add timestep statistics
        total_timesteps = cumulative_timesteps[-1]
        avg_steps_per_ep = total_timesteps / len(scores)
        timestep_stats = f'Total Timesteps: {total_timesteps:,}\nAvg Steps/Episode: {avg_steps_per_ep:.0f}'
        ax2.text(0.02, 0.98, timestep_stats, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = checkpoint_path.replace('.pkl', '_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Graph saved to: {output_path}")
    
    plt.show()


def plot_all_checkpoints_comparison(checkpoint_dir='train'):
    """Plot comparison of multiple checkpoints - using LATEST checkpoint for full history"""
    
    # Find all training_state files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'training_state_ep*.pkl'))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}/")
        return
    
    # Sort by episode number
    checkpoint_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pkl')[0]))
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print("Detecting training sessions (restarts)...")
    
    # Detect training sessions by finding where data count decreases (restart)
    sessions = []
    current_session = []
    prev_scores_len = 0
    
    for checkpoint_path in checkpoint_files:
        with open(checkpoint_path, 'rb') as f:
            training_state = pickle.load(f)
        
        checkpoint_episode = training_state['episode']
        num_scores = len(training_state['scores'])
        
        # If scores decreased, it's a new session (restart)
        if num_scores < prev_scores_len and len(current_session) > 0:
            sessions.append(current_session[-1])  # Save last checkpoint of previous session
            current_session = []
            print(f"  📍 Session break detected at ep {checkpoint_episode}")
        
        current_session.append((checkpoint_path, checkpoint_episode, num_scores))
        prev_scores_len = num_scores
    
    # Add last session
    if current_session:
        sessions.append(current_session[-1])
    
    print(f"\n✅ Found {len(sessions)} training sessions")
    
    # Load and merge all sessions
    all_scores = []
    all_done_timesteps = []
    
    for i, (checkpoint_path, checkpoint_episode, num_scores) in enumerate(sessions):
        print(f"\n  Session {i+1}: Loading {checkpoint_path}")
        print(f"    Episode: {checkpoint_episode}, Contains: {num_scores} episodes")
        
        with open(checkpoint_path, 'rb') as f:
            training_state = pickle.load(f)
        
        session_scores = training_state['scores']
        session_timesteps = training_state['done_timesteps']
        
        all_scores.extend(session_scores)
        all_done_timesteps.extend(session_timesteps)
        print(f"    ✅ Added {len(session_scores)} episodes (total now: {len(all_scores)})")
    
    scores = all_scores
    done_timesteps = all_done_timesteps
    episode = len(scores)
    
    print(f"\n🎉 Merged total: {len(scores)} episodes from {len(sessions)} sessions")
    
    # Calculate cumulative timesteps
    cumulative_timesteps = np.cumsum(done_timesteps)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ===== PLOT 1: Score vs Episodes =====
    N = 100
    if len(scores) >= N:
        smoothed_scores = np.convolve(np.array(scores), np.ones((N,)) / N, mode='valid')
        episode_indices = np.arange(N-1, len(scores))
        
        ax1.plot(episode_indices, smoothed_scores, linewidth=2, color='blue')
        ax1.set_xlabel('Episode #', fontsize=12)
        ax1.set_ylabel('Score (100-episode mean)', fontsize=12)
        ax1.set_title(f'DQN Training Progress - Full History (0 to {episode} episodes)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        max_score = max(scores)
        max_episode = scores.index(max_score)
        avg_last_100 = np.mean(scores[-100:])
        stats_text = f'Max Score: {max_score:.0f} (Ep {max_episode})\nAvg (last 100): {avg_last_100:.2f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== PLOT 2: Score vs Timesteps =====
    if len(scores) >= N:
        smoothed_timesteps = np.convolve(cumulative_timesteps, np.ones((N,)) / N, mode='valid')
        
        ax2.plot(smoothed_timesteps, smoothed_scores, linewidth=2, color='green')
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Score (100-episode mean)', fontsize=12)
        ax2.set_title('DQN Training Progress - Score vs Timesteps', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add timestep statistics
        total_timesteps = cumulative_timesteps[-1]
        avg_steps_per_ep = total_timesteps / len(scores)
        timestep_stats = f'Total Timesteps: {total_timesteps:,}\nAvg Steps/Episode: {avg_steps_per_ep:.0f}'
        ax2.text(0.02, 0.98, timestep_stats, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = os.path.join(checkpoint_dir, 'full_training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Full history graph saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Plot specific checkpoint
        checkpoint_path = sys.argv[1]
        if os.path.exists(checkpoint_path):
            plot_from_checkpoint(checkpoint_path)
        else:
            print(f"❌ File not found: {checkpoint_path}")
    else:
        # Plot all checkpoints comparison
        print("Usage: python plot_checkpoint.py <checkpoint_path>")
        print("Example: python plot_checkpoint.py train/training_state_ep1000.pkl")
        print("\nOr run without arguments to compare all checkpoints:")
        
        response = input("\nPlot comparison of all checkpoints? (y/n): ")
        if response.lower() == 'y':
            plot_all_checkpoints_comparison()
        else:
            # Find and list available checkpoints
            checkpoint_files = glob.glob('train*/training_state_ep*.pkl')
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pkl')[0]))
                print("\nAvailable checkpoints:")
                for cp in checkpoint_files[-10:]:  # Show last 10
                    print(f"  - {cp}")
                print("\nRun: python plot_checkpoint.py <checkpoint_path>")
