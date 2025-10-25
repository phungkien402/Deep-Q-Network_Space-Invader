"""
DQN Training cho Space Invaders - Đồ Án Tốt Nghiệp
===================================================
Cài đặt Double DQN với các tính năng chính:
- Frame Stacking: Xếp chồng 4 khung hình liên tiếp để hiểu chuyển động
- Frame Skipping: Lặp lại mỗi hành động 4 lần (chuẩn DeepMind)
- Reward Shaping: Điều chỉnh phần thưởng để hướng dẫn agent học tốt hơn
- Experience Replay: Lưu và học lại từ kinh nghiệm quá khứ
- Target Network: Mạng riêng biệt để ổn định Q-values
- Epsilon-Greedy: Cân bằng giữa khám phá và khai thác

Tác giả: [Tên của bạn]
Ngày: Tháng 10/2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque
import gymnasium as gym
import ale_py
from gymnasium import wrappers
import torch
import signal
import sys
import random
import time
from agent import DQNAgent, FrameStack, DEVICE

# Đăng ký môi trường ALE (Arcade Learning Environment)
gym.register_envs(ale_py)

# =========================================================================
# KHỞI TẠO RANDOM SEED: Đảm bảo mỗi lần chạy có kết quả khác nhau
# =========================================================================
# Dùng timestamp để tạo seed khác nhau mỗi lần train
# Điều này ngăn hành vi deterministic và đảm bảo đa dạng trong training
random_seed = int(time.time() * 1000) % 2**32
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
print(f"🎲 Random seed: {random_seed}")

# =========================================================================
# THÔNG TIN HỆ THỐNG: Kiểm tra GPU có sẵn không
# =========================================================================
print("=" * 60)
print("THÔNG TIN HỆ THỐNG")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("⚠️  CẢNH BÁO: KHÔNG CÓ CUDA - TRAINING TRÊN CPU!")
print("=" * 60)
print()


# =========================================================================
# HÀM TRAINING
# =========================================================================
def train(n_episodes=200,
          max_t=10000,
          eps_start=1.0,  # Bắt đầu với 100% khám phá
          eps_end=0.01,   # Tối thiểu 1% khám phá để tránh bị stuck
          eps_decay=0.995,  # Tốc độ giảm epsilon (~5% sau 500 episodes)
          start_episode=0,  # Episode bắt đầu (để tiếp tục training)
          update_every=4,  # Học mỗi 4 bước (lựa chọn của DeepMind)
          hyperparams=None):  # Tham số cho việc lưu checkpoint
    """
    Training agent Deep Q-Learning để chơi Space Invaders
    ---
    Tham số
    =======
        n_episodes (int): Số lượng episodes tối đa để train
        max_t (int): Số timesteps tối đa mỗi episode
        eps_start (float): Giá trị epsilon bắt đầu (khám phá)
        eps_end (float): Giá trị epsilon tối thiểu
        eps_decay (float): Hệ số giảm epsilon mỗi episode
        start_episode (int): Episode bắt đầu (khi tiếp tục training)
    Trả về
    ======
        scores: Danh sách điểm số của mỗi episode
        done_timesteps: Danh sách số bước của mỗi episode
    """
    # Lưu điểm số của mỗi episode
    scores = []

    # Lưu số timesteps của mỗi episode khi game kết thúc
    done_timesteps = []

    # 100 điểm gần nhất dùng để tính điểm trung bình
    scores_window = deque(maxlen=100)
    
    # Theo dõi loss trong quá trình training
    losses = []
    
    eps = eps_start
    
    # =========================================================================
    # XỬ LÝ CTRL+C: Hiển thị biểu đồ tiến trình khi dừng training
    # =========================================================================
    def signal_handler(sig, frame):
        print('\n\n🛑 Training bị dừng bởi người dùng (Ctrl+C)')
        print(f'📊 Đang vẽ biểu đồ cho {len(scores)} episodes...')
        
        if len(scores) >= 100:
            # Vẽ running mean (trung bình trượt)
            N = 100
            cumulative_timesteps = np.cumsum(done_timesteps)
            smoothed_scores = np.convolve(np.array(scores), np.ones((N,)) / N, mode='valid')
            smoothed_timesteps = np.convolve(cumulative_timesteps, np.ones((N,)) / N, mode='valid')
            
            fig = plt.figure()
            plt.plot(smoothed_timesteps, smoothed_scores)
            plt.ylabel('Điểm Số')
            plt.xlabel('Timesteps')
            plt.title('DQN Training - Trung Bình 100 Episodes (Bị Dừng)')
            plt.show()
        else:
            print(f'⚠️ Chưa đủ episodes ({len(scores)}) để tính trung bình 100 episodes')
        
        sys.exit(0)
    
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # =========================================================================
    # FRAME STACKING: Xếp chồng 4 khung hình để cung cấp thông tin thời gian
    # =========================================================================
    # Xếp chồng 4 khung hình liên tiếp để agent có thể nhận biết chuyển động và vận tốc
    # Điều này quan trọng cho game Atari vì 1 khung hình không thể hiện chuyển động
    frame_stack = FrameStack(num_frames=4)
    
    # Debug: Kiểm tra việc random có hoạt động không
    if start_episode == 0:
        print(f"🔍 5 giá trị random đầu tiên: {[np.random.rand() for _ in range(5)]}")
    
    # =========================================================================
    # VÒNG LẶP TRAINING CHÍNH: Lặp qua các episodes
    # =========================================================================
    for i_episode in range(start_episode + 1, start_episode + n_episodes + 1):
        # Reset môi trường với seed ngẫu nhiên để đa dạng
        episode_seed = np.random.randint(0, 2**31 - 1)
        raw_state, info = env.reset(seed=episode_seed)
        
        # Khởi tạo frame stack với observation đầu tiên
        processed_frame = agent.preprocess_state(raw_state).cpu().numpy().squeeze()
        frame_stack.reset(processed_frame)
        state = torch.from_numpy(np.stack(frame_stack.frames, axis=0)).float().unsqueeze(0).to(DEVICE)
        
        # Biến theo dõi episode
        score = 0  # Tổng phần thưởng của episode này
        episode_experiences = []  # Lưu tất cả experiences để replay
        prev_lives = info.get('lives', 3)  # Theo dõi số mạng để phạt khi chết
        prev_action = None  # Theo dõi đa dạng hành động
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Phân phối hành động
        episode_losses = []  # Theo dõi training loss
        
        # =====================================================================
        # VÒNG LẶP EPISODE: Tương tác với môi trường và học
        # =====================================================================
        for timestep in range(max_t):
            # CHỌN HÀNH ĐỘNG: Epsilon-greedy policy
            action = agent.act(state, eps)
            
            # THỰC HIỆN HÀNH ĐỘNG: Bước trong môi trường
            raw_next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # XỬ LÝ OBSERVATION: Áp dụng frame stacking
            processed_next_frame = agent.preprocess_state(raw_next_state).cpu().numpy().squeeze()
            frame_stack.add(processed_next_frame)
            next_state = torch.from_numpy(np.stack(frame_stack.frames, axis=0)).float().unsqueeze(0).to(DEVICE)
            
            # Theo dõi phân phối hành động để phân tích
            action_counts[action] += 1
            
            # =====================================================================
            # REWARD SHAPING: Thiết kế phần thưởng đa mục tiêu
            # =====================================================================
            # Phần thưởng cơ bản từ môi trường (điểm từ việc tiêu diệt kẻ địch)
            shaped_reward = reward
            current_lives = info.get('lives', 0)
            
            # 1. PHẠT MẤT MẠNG: Tín hiệu âm mạnh khi agent chết
            #    Giúp agent học cách tránh đạn địch
            if current_lives < prev_lives:
                shaped_reward -= 5.0  # Phạt đáng kể để ưu tiên sống sót
                print(f"    💔 Mất mạng! Mạng còn lại: {current_lives} (Episode {i_episode}, Bước {timestep})")
            
            # 2. THƯỞNG SỐNG SÓT: Phần thưởng nhỏ cho việc sống sót
            #    Khuyến khích episodes dài hơn và nhiều cơ hội học hơn
            if not done:
                shaped_reward += 0.01
            
            # 3. THƯỞNG BẮN: Khuyến khích chơi tấn công
            #    Space Invaders cần bắn để ghi điểm, điều này hướng dẫn agent
            if action in [1, 4, 5]:  # FIRE, RIGHTFIRE, LEFTFIRE
                shaped_reward += 0.05
            
            # 4. THƯỞNG DI CHUYỂN: Khuyến khích tìm vị trí chiến thuật
            #    Agent cần né tránh và di chuyển để tránh đạn địch
            if action in [2, 3]:  # RIGHT, LEFT (di chuyển ngang)
                shaped_reward += 0.03
            
            # 5. PHẠT KHÔNG LÀM GÌ: Không khuyến khích đợi quá nhiều
            #    Ngăn agent học các chiến lược thụ động
            if action == 0:  # NOOP (không làm gì)
                shaped_reward -= 0.01

            prev_lives = current_lives
            prev_action = action
            
            # =====================================================================
            # EXPERIENCE REPLAY: Store and learn from experience
            # =====================================================================
            # Store experience tuple (s, a, r, s', done) for later replay
            episode_experiences.append((state, action, shaped_reward, next_state, done))
            
            # LEARN: Update Q-network using experience replay
            loss = agent.step(state, action, shaped_reward, next_state, done)
            if loss is not None:
                episode_losses.append(loss)
            
            # ⚡ Aggressive memory cleanup for 4GB GPU
            if timestep % 100 == 0:
                torch.cuda.empty_cache()
            
            state = next_state
            score += reward  # Score vẫn dùng reward gốc để đánh giá
            if done:
                done_timesteps.append(timestep)
                break

        # Append score AFTER episode ends
        scores_window.append(score)
        scores.append(score)
        
        # Calculate average score
        avg_score = np.mean(scores_window) if len(scores_window) > 0 else 0
        
        # Calculate average loss for this episode
        avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0
        if len(episode_losses) > 0:
            losses.append(avg_loss)
        
        # Print action distribution every 50 episodes
        if i_episode % 1 == 0:
            left_usage = action_counts[3]
            right_usage = action_counts[2]
            right_fire_usage = action_counts[4]
            left_fire_usage = action_counts[5]
            fire_usage = action_counts[1]
            noop_usage = action_counts[0]
            total_actions = sum(action_counts.values())

            print(f"    📊 Action stats: LEFT={left_usage}, RIGHT={right_usage}, LEFT_FIRE={left_fire_usage}, RIGHT_FIRE={right_fire_usage}, FIRE={fire_usage}, NOOP={noop_usage}, TOTAL={total_actions}")

        # Decrease epsilon
        eps = max(eps * eps_decay, eps_end)
        
        # Print episode summary with loss and total timesteps
        loss_str = f'Loss: {avg_loss:.4f}' if avg_loss > 0 else 'Loss: N/A'
        total_steps = agent.total_steps
        warm_up_status = f' 🔥 LEARNING!' if total_steps >= agent.learning_starts else f' ⏳ Warm-up: {total_steps}/{agent.learning_starts}'
        print('Episode {}\tScore: {:.2f}\tAvg: {:.2f}\t{}\tEps: {:.3f}\tSteps: {}{}'.format(
            i_episode, score, avg_score, loss_str, eps, total_steps, warm_up_status))
        
        # ⚡ Cleanup episode data and GPU cache (4GB GPU)
        del episode_experiences, episode_losses
        torch.cuda.empty_cache()
        
        # ⚡ Aggressive GPU memory cleanup every 10 episodes
        if i_episode % 10 == 0:
            # Clear optimizer state to free memory
            agent.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # Print memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"    🧹 GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        # Save model every SAVE_EVERY episodes
        if i_episode % SAVE_EVERY == 0:
            # Save model weights với tên riêng theo episode
            checkpoint_name = f'model_ep{i_episode}.pth'
            torch.save(agent.qnetwork_local.state_dict(),
                       SAVE_DIR + checkpoint_name)
            print(f'💾 Checkpoint saved: {checkpoint_name}')
            
            # ⚡ Save COMPREHENSIVE training state for thesis reporting
            training_state = {
                # Training progress
                'epsilon': eps,
                'episode': i_episode,
                'total_steps': agent.total_steps,
                
                # Performance metrics
                'scores': scores,
                'done_timesteps': done_timesteps,
                'losses': losses,  # Average loss per episode
                
                # Statistics for reporting
                'avg_score_last_100': np.mean(scores_window) if len(scores_window) > 0 else 0,
                'max_score': np.max(scores) if len(scores) > 0 else 0,
                'min_score': np.min(scores) if len(scores) > 0 else 0,
                'avg_loss': np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if len(losses) > 0 else 0),
                
                # Hyperparameters (for reproducibility)
                'hyperparameters': {
                    'buffer_size': hyperparams['buffer_size'] if hyperparams else 'unknown',
                    'batch_size': agent.batch_size,
                    'gamma': agent.gamma,
                    'lr': hyperparams['lr'] if hyperparams else 'unknown',
                    'tau': agent.tau,
                    'update_every': agent.update_every,
                    'learning_starts': agent.learning_starts,
                    'eps_start': eps_start,
                    'eps_end': eps_end,
                    'eps_decay': eps_decay,
                    'frameskip': 4  # Document this important parameter
                },
                
                # Timestamps
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'random_seed': random_seed
            }
            state_name = f'training_state_ep{i_episode}.pkl'
            with open(SAVE_DIR + state_name, 'wb') as fp:
                pickle.dump(training_state, fp)
            
            print(f'📊 Stats: Avg={training_state["avg_score_last_100"]:.2f}, Max={training_state["max_score"]:.0f}, Avg Loss={training_state["avg_loss"]:.4f}')

    # save the final network
    torch.save(agent.qnetwork_local.state_dict(), SAVE_DIR + 'model.pth')
    
    # ⚡ Save COMPREHENSIVE final training state
    training_state = {
        # Training progress
        'epsilon': eps,
        'episode': start_episode + n_episodes,
        'total_steps': agent.total_steps,
        
        # Performance metrics
        'scores': scores,
        'done_timesteps': done_timesteps,
        'losses': losses,
        
        # Statistics for reporting
        'avg_score_last_100': np.mean(scores_window) if len(scores_window) > 0 else 0,
        'max_score': np.max(scores) if len(scores) > 0 else 0,
        'min_score': np.min(scores) if len(scores) > 0 else 0,
        'avg_loss': np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if len(losses) > 0 else 0),
        
        # Hyperparameters
        'hyperparameters': {
            'buffer_size': hyperparams['buffer_size'] if hyperparams else 'unknown',
            'batch_size': agent.batch_size,
            'gamma': agent.gamma,
            'lr': hyperparams['lr'] if hyperparams else 'unknown',
            'tau': agent.tau,
            'update_every': agent.update_every,
            'learning_starts': agent.learning_starts,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay,
            'frameskip': 4
        },
        
        # Timestamps
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'random_seed': random_seed
    }
    with open(SAVE_DIR + 'training_state.pkl', 'wb') as fp:
        pickle.dump(training_state, fp)
    
    print(f'\n{"="*60}')
    print(f'📊 FINAL TRAINING STATISTICS')
    print(f'{"="*60}')
    print(f'Total Episodes: {start_episode + n_episodes}')
    print(f'Total Steps: {agent.total_steps:,}')
    print(f'Avg Score (last 100): {training_state["avg_score_last_100"]:.2f}')
    print(f'Max Score: {training_state["max_score"]:.0f}')
    print(f'Min Score: {training_state["min_score"]:.0f}')
    print(f'Avg Loss: {training_state["avg_loss"]:.4f}')
    print(f'Final Epsilon: {eps:.4f}')
    print(f'{"="*60}\n')

    # save the final scores (for backward compatibility)
    with open(SAVE_DIR + 'scores', 'wb') as fp:
        pickle.dump(scores, fp)

    # save the done timesteps (for backward compatibility)
    with open(SAVE_DIR + 'dones', 'wb') as fp:
        pickle.dump(done_timesteps, fp)

    return scores, done_timesteps


def test(env, trained_agent, n_games=5, n_steps_per_game=10000):
    # ⚡ FRAME STACKING for test mode
    frame_stack = FrameStack(num_frames=4)
    
    for game in range(n_games):
        env = wrappers.RecordVideo(env,
                               "./test/game-{}".format(game),
                               episode_trigger=lambda x: True)

        raw_observation, _ = env.reset()
        
        # Initialize frame stack
        processed_frame = trained_agent.preprocess_state(raw_observation).cpu().numpy().squeeze()
        frame_stack.reset(processed_frame)
        observation = torch.from_numpy(np.stack(frame_stack.frames, axis=0)).float().unsqueeze(0).to(DEVICE)
        
        score = 0
        action_counts = [0] * env.action_space.n  # Track action distribution
        debug_step = 0
        for step in range(n_steps_per_game):
            # Debug first few steps
            debug = (debug_step < 3)
            action = trained_agent.act(observation, eps=0.0, debug=debug)  # eps=0 để không random
            action_counts[action] += 1
            
            if debug_step < 3:
                print(f"  Step {debug_step}: action={action}")
                debug_step += 1
            
            raw_observation, reward, done, truncated, info = env.step(action)
            
            # Process next frame with stacking
            processed_frame = trained_agent.preprocess_state(raw_observation).cpu().numpy().squeeze()
            frame_stack.add(processed_frame)
            observation = torch.from_numpy(np.stack(frame_stack.frames, axis=0)).float().unsqueeze(0).to(DEVICE)
            
            score += reward
            if done or truncated:
                print('GAME-{} OVER! score={}'.format(game, score))
                print(f'  Action distribution: {action_counts}')
                break
        env.close()


# Agent was trained on GPU in colab.
# The files presented in train folder are those colab
# TODO
# - Encapsulate the training data into a trainloader to avoid GPU runtime error

if __name__ == '__main__':
    TRAIN = True  # ⚡ TRAIN mode with frame stacking!
    BUFFER_SIZE = 20000  # ⚡ REDUCED for 4GB GPU (lower for safety)
    BATCH_SIZE = 32  # ⚡ REDUCED to 24 for 4GB GPU (was 32)
    GAMMA = 0.90  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 0.0001  # learning rate 1e-4 (document recommended)
    UPDATE_EVERY = 16  # Learn every 16 steps (less frequent for 4GB GPU)
    SAVE_EVERY = 50  # ⚡ Save thường xuyên hơn (mỗi 50 episodes)
    MAX_TIMESTEPS = 10000  # max timesteps mỗi episode
    N_EPISODES = 1000  # ⚡ Train thêm 2000 episodes
    SAVE_DIR = "./train/"  # Lưu vào folder train duy nhất
    LOAD_MODEL = True   # ⚡ TRUE: Continue training from ep2000 with epsilon=0.01

    if LOAD_MODEL:
        LEARNING_STARTS = 0
    else:
        LEARNING_STARTS = 10000  # ⚡ NO WARM-UP when continuing training (was 10000)

    # Create environment with render_mode for recording videos in test mode
    # ⚡ FRAMESKIP: Repeat each action for 4 frames (DeepMind standard)
    # This makes agent move faster (like in demo videos) and speeds up training 4x
    if TRAIN:
        env = gym.make('SpaceInvaders-v4', frameskip=4)
    else:
        env = gym.make('SpaceInvaders-v4', render_mode='rgb_array', frameskip=4)

    if TRAIN:
        # init agent
        agent = DQNAgent(state_size=4,
                         action_size=env.action_space.n,
                         seed=0,
                         lr=LR,
                         gamma=GAMMA,
                         tau=TAU,
                         buffer_size=BUFFER_SIZE,
                         batch_size=BATCH_SIZE,
                         update_every=UPDATE_EVERY,
                         learning_starts=LEARNING_STARTS)  # SB3: Warm-up period
        
        # Load model đã train trước (nếu có)
        loaded_epsilon = None
        loaded_episode = 0
        if LOAD_MODEL:
            try:
                # ⚡ AUTO-FIND latest checkpoint
                import glob
                checkpoint_files = glob.glob(SAVE_DIR + 'model_ep*.pth')
                if checkpoint_files:
                    # Sort by episode number (extract from filename)
                    checkpoint_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
                    latest_checkpoint = checkpoint_files[-1]
                    episode_num = int(latest_checkpoint.split('_ep')[1].split('.pth')[0])
                    
                    checkpoint_path = latest_checkpoint
                    state_path = SAVE_DIR + f'training_state_ep{episode_num}.pkl'
                else:
                    # Fallback to generic files
                    checkpoint_path = SAVE_DIR + 'model.pth'
                    state_path = SAVE_DIR + 'training_state.pkl'
                
                agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
                agent.qnetwork_target.load_state_dict(torch.load(checkpoint_path))
                print(f"✅ Loaded model from {checkpoint_path}")
                
                # Load training state (epsilon, episode count, etc.)
                try:
                    with open(state_path, 'rb') as fp:
                        training_state = pickle.load(fp)
                        loaded_epsilon = training_state.get('epsilon', None)
                        loaded_episode = training_state.get('episode', 0)
                        
                        # ⚡ ASK USER: Adjust epsilon?
                        print(f"\n{'='*60}")
                        print(f"📊 CURRENT TRAINING STATE")
                        print(f"{'='*60}")
                        print(f"Episode: {loaded_episode}")
                        print(f"Current Epsilon: {loaded_epsilon:.4f}")
                        print(f"Avg Score (last 100): {np.mean(training_state['scores'][-100:]):.2f}")
                        print(f"{'='*60}\n")
                        
                        response = input("🔍 Change epsilon? (y/n) [default: n]: ").strip().lower()
                        
                        if response == 'y':
                            try:
                                new_epsilon = float(input(f"Enter new epsilon value (current: {loaded_epsilon:.4f}): "))
                                if 0 <= new_epsilon <= 1:
                                    loaded_epsilon = new_epsilon
                                    print(f"✅ Epsilon set to: {new_epsilon:.4f}")
                                else:
                                    print(f"⚠️ Invalid epsilon (must be 0-1), keeping {loaded_epsilon:.4f}")
                            except:
                                print(f"⚠️ Invalid input, keeping epsilon at {loaded_epsilon:.4f}")
                        else:
                            print(f"✅ Keeping epsilon at {loaded_epsilon:.4f}")
                        
                        # ⚡ CRITICAL: Restore total_steps from saved state (not estimate!)
                        saved_steps = training_state.get('total_steps', None)
                        if saved_steps is not None:
                            agent.total_steps = saved_steps
                            print(f"✅ Restored total_steps to: {saved_steps:,}")
                        else:
                            # Fallback to estimate if old checkpoint without total_steps
                            estimated_steps = loaded_episode * 1000
                            agent.total_steps = estimated_steps
                            print(f"⚠️ Estimated total_steps to: {estimated_steps:,} (old checkpoint)")
                        
                        if loaded_epsilon:
                            print(f"✅ Epsilon set to: {loaded_epsilon:.4f} (was {training_state.get('epsilon', 0):.4f})")
                        if loaded_episode:
                            print(f"✅ Continuing from episode: {loaded_episode}")
                except:
                    print("⚠️ No training state found, using default epsilon")
            except:
                print("⚠️ No checkpoint found, starting from scratch")
        
        # Prepare hyperparameters dict for checkpoint saving
        hyperparams_dict = {
            'buffer_size': BUFFER_SIZE,
            'batch_size': BATCH_SIZE,
            'gamma': GAMMA,
            'lr': LR,
            'tau': TAU,
            'update_every': UPDATE_EVERY,
            'learning_starts': LEARNING_STARTS
        }
        
        # train and get the scores
        # Pass loaded epsilon and episode to continue from where we left off
        if loaded_epsilon is not None:
            scores, done_timesteps = train(n_episodes=N_EPISODES, max_t=MAX_TIMESTEPS, 
                          eps_start=loaded_epsilon, start_episode=loaded_episode,
                          hyperparams=hyperparams_dict)
        else:
            scores, done_timesteps = train(n_episodes=N_EPISODES, max_t=MAX_TIMESTEPS,
                          hyperparams=hyperparams_dict)

        # plot the running mean of scores
        N = 100  # running mean window
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # Calculate cumulative timesteps for each episode
        cumulative_timesteps = np.cumsum(done_timesteps)
        
        # Apply running mean to both scores and timesteps
        smoothed_scores = np.convolve(np.array(scores), np.ones((N, )) / N, mode='valid')
        smoothed_timesteps = np.convolve(cumulative_timesteps, np.ones((N, )) / N, mode='valid')
        
        plt.plot(smoothed_timesteps, smoothed_scores)
        plt.ylabel('Score')
        plt.xlabel('Timesteps')
        plt.title('DQN Training - 100-Episode Running Mean')
        plt.show()
    else:
        N_GAMES = 5
        N_STEPS_PER_GAME = 10000

        # init a new agent
        trained_agent = DQNAgent(state_size=4,
                                 action_size=env.action_space.n,
                                 seed=0)

        # ⚠️ COMMENTED OUT: Old model incompatible with frame stacking (1 channel → 4 channels)
        # # replace the weights with the trained weights
        # # Load latest checkpoint (model_ep550)
        # model_path = "./train/model/model_ep550.pth"
        # print(f"Loading model from {model_path}...")
        # trained_agent.qnetwork_local.load_state_dict(
        #     torch.load(model_path))
        
        # # Verify model loaded correctly by checking some weights
        # print("✅ Model loaded. First layer weight sum:", 
        #       trained_agent.qnetwork_local.conv1.weight.sum().item())

        # enable inference mode
        trained_agent.qnetwork_local.eval()

        # test and save results to disk
        test(env, trained_agent)
