import numpy as np
import time
from environment import SimpleMaze, MazeVisualizer


def train_qlearning_agent(episodes=1000, visualize=False):
    """
    Train Q-learning agent.

    Args:
        episodes (int): Number of training episodes
        visualize (bool): Whether to visualize training

    Returns:
        agent: Trained agent
        rewards_per_episode (list): Rewards for each episode
        steps_per_episode (list): Steps taken in each episode
    """
    # Import here to avoid circular imports
    from q_learning import QLearningAgent

    env = SimpleMaze()
    agent = QLearningAgent()
    rewards_per_episode = []
    steps_per_episode = []

    visualizer = None
    if visualize:
        visualizer = MazeVisualizer(env)

    print(f"Training Q-Learning agent for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:  # Max 100 steps per episode
            # Choose action
            valid_actions = env.get_valid_actions(state)
            action = agent.get_action(state, valid_actions)

            # Take action
            next_state, reward, done = env.step(action)
            next_valid_actions = env.get_valid_actions(next_state)

            # Learn from experience
            agent.learn(state, action, reward, next_state, next_valid_actions)

            # Update state and metrics
            state = next_state
            total_reward += reward
            steps += 1

            # Update visualization
            if visualize:
                visualizer.update_agent(state, episode=episode, step=steps)
                time.sleep(0.1)  # Control animation speed

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f'Episode {episode}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}')

    if visualizer:
        visualizer.close()

    return agent, rewards_per_episode, steps_per_episode


def train_sarsa_agent(episodes=1000, visualize=False):
    """
    Train SARSA agent.

    Args:
        episodes (int): Number of training episodes
        visualize (bool): Whether to visualize training

    Returns:
        agent: Trained agent
        rewards_per_episode (list): Rewards for each episode
        steps_per_episode (list): Steps taken in each episode
    """
    # Import here to avoid circular imports
    from sarsa import SARSAAgent

    env = SimpleMaze()
    agent = SARSAAgent()
    rewards_per_episode = []
    steps_per_episode = []

    visualizer = None
    if visualize:
        visualizer = MazeVisualizer(env)

    print(f"Training SARSA agent for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # For SARSA, we need to choose the first action before the loop
        valid_actions = env.get_valid_actions(state)
        action = agent.get_action(state, valid_actions)

        while not done and steps < 100:  # Max 100 steps per episode
            # Take action
            next_state, reward, done = env.step(action)
            next_valid_actions = env.get_valid_actions(next_state)

            # Choose next action (on-policy)
            if not done:
                next_action = agent.get_action(next_state, next_valid_actions)
            else:
                next_action = None  # Terminal state

            # Learn from experience
            agent.learn(state, action, reward, next_state, next_action)

            # Update state and action
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            # Update visualization
            if visualize:
                visualizer.update_agent(state, episode=episode, step=steps)
                time.sleep(0.1)  # Control animation speed

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f'Episode {episode}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}')

    if visualizer:
        visualizer.close()

    return agent, rewards_per_episode, steps_per_episode


def test_agent(agent, episodes=5, visualize=True):
    """
    Test a trained agent.

    Args:
        agent: Trained agent (Q-learning or SARSA)
        episodes (int): Number of test episodes
        visualize (bool): Whether to visualize testing
    """
    env = SimpleMaze()

    # Set epsilon to 0 for greedy testing
    original_epsilon = agent.epsilon
    agent.set_epsilon(0)

    visualizer = None
    if visualize:
        visualizer = MazeVisualizer(env)

    print(f"\nTesting agent for {episodes} episodes...")
    print("Agent uses greedy policy (epsilon=0) for testing\n")

    success_count = 0
    total_steps = 0

    for episode in range(episodes):
        state = env.reset()
        steps = 0
        done = False
        path = [state]

        print(f'Test Episode {episode + 1}:')

        if visualize:
            visualizer.update_agent(state, episode=episode + 1, step=0)
            time.sleep(0.5)

        while not done and steps < 50:
            valid_actions = env.get_valid_actions(state)
            # Choose best action (greedy)
            q_values = agent.q_table[state]
            valid_q_values = {action: q_values[action] for action in valid_actions}
            action = max(valid_q_values.keys(), key=lambda x: valid_q_values[x])

            next_state, reward, done = env.step(action)
            path.append(next_state)

            print(f'  Step {steps + 1}: Position {state} -> Action {action} -> Position {next_state}, Reward: {reward}')

            state = next_state
            steps += 1

            if visualize:
                visualizer.update_agent(state, episode=episode + 1, step=steps + 1)
                time.sleep(0.5)

            if done:
                if reward > 0:
                    print(f'  SUCCESS! Reached target in {steps + 1} steps!')
                    success_count += 1
                else:
                    print(f'  FAILED! Hit a trap after {steps + 1} steps.')
                total_steps += steps
                break

    # Restore original epsilon
    agent.set_epsilon(original_epsilon)

    # Print summary
    if visualizer:
        visualizer.close()

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    if success_count > 0:
        print(f"Average steps to success: {total_steps/success_count:.1f}")
    print("="*50)


def compare_algorithms(episodes=500):
    """
    Compare Q-learning and SARSA algorithms.

    Args:
        episodes (int): Number of training episodes
    """
    print("\n" + "="*50)
    print("COMPARING Q-LEARNING vs SARSA")
    print("="*50)

    # Train Q-learning
    print("\n1. Training Q-Learning Agent...")
    q_agent, q_rewards, q_steps = train_qlearning_agent(episodes=episodes, visualize=False)

    # Train SARSA
    print("\n2. Training SARSA Agent...")
    s_agent, s_rewards, s_steps = train_sarsa_agent(episodes=episodes, visualize=False)

    # Plot learning curves
    plt.figure(figsize=(12, 5))

    # Rewards
    plt.subplot(1, 2, 1)
    plt.plot(q_rewards, label='Q-Learning', alpha=0.7)
    plt.plot(s_rewards, label='SARSA', alpha=0.7)
    # Smooth curves with moving average
    window = min(50, episodes//10)
    if window > 1:
        q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        s_smooth = np.convolve(s_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, episodes), q_smooth, label='Q-Learning (smooth)', linewidth=2)
        plt.plot(range(window-1, episodes), s_smooth, label='SARSA (smooth)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curves - Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Steps
    plt.subplot(1, 2, 2)
    plt.plot(q_steps, label='Q-Learning', alpha=0.7)
    plt.plot(s_steps, label='SARSA', alpha=0.7)
    if window > 1:
        q_smooth_steps = np.convolve(q_steps, np.ones(window)/window, mode='valid')
        s_smooth_steps = np.convolve(s_steps, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, episodes), q_smooth_steps, label='Q-Learning (smooth)', linewidth=2)
        plt.plot(range(window-1, episodes), s_smooth_steps, label='SARSA (smooth)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Steps to Completion')
    plt.title('Learning Curves - Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final performance
    print("\n" + "="*50)
    print("FINAL PERFORMANCE (Last 100 Episodes)")
    print("="*50)
    print(f"Q-Learning - Average Reward: {np.mean(q_rewards[-100:]):.2f}, "
          f"Average Steps: {np.mean(q_steps[-100:]):.1f}")
    print(f"SARSA - Average Reward: {np.mean(s_rewards[-100:]):.2f}, "
          f"Average Steps: {np.mean(s_steps[-100:]):.1f}")
    print("="*50)

    # Test both agents
    print("\n3. Testing Q-Learning Agent...")
    test_agent(q_agent, episodes=3, visualize=True)

    print("\n4. Testing SARSA Agent...")
    test_agent(s_agent, episodes=3, visualize=True)

    return q_agent, s_agent


def get_user_choice():
    """Get user's choice for training mode"""
    print("Please choose an option:")
    print("1. Train and test Q-Learning agent")
    print("2. Train and test SARSA agent")
    print("3. Compare Q-Learning and SARSA")
    print("4. Fast training only (no visualization)")
    print("5. Exit")

    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


def main():
    """Main function to run training and testing"""
    print("="*50)
    print("REINFORCEMENT LEARNING - MAZE NAVIGATION")
    print("="*50)

    while True:
        choice = get_user_choice()

        if choice == '5':  # Exit
            print("Goodbye!")
            break

        # Ask about visualization for single agent training
        visualize = False
        if choice in ['1', '2']:
            viz_choice = input("Visualize training? (y/n): ").strip().lower()
            visualize = viz_choice == 'y'

        # Get number of episodes
        episodes = input("Number of training episodes (default 500): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 500

        if choice == '1':  # Q-Learning
            agent, rewards, steps = train_qlearning_agent(episodes=episodes, visualize=visualize)
            print('\nTraining completed!')
            print(f'Average reward (last 100): {np.mean(rewards[-100:]):.2f}')
            print(f'Average steps (last 100): {np.mean(steps[-100:]):.1f}')
            test_agent(agent, episodes=3, visualize=True)

        elif choice == '2':  # SARSA
            agent, rewards, steps = train_sarsa_agent(episodes=episodes, visualize=visualize)
            print('\nTraining completed!')
            print(f'Average reward (last 100): {np.mean(rewards[-100:]):.2f}')
            print(f'Average steps (last 100): {np.mean(steps[-100:]):.1f}')
            test_agent(agent, episodes=3, visualize=True)

        elif choice == '3':  # Compare
            compare_algorithms(episodes=episodes)

        elif choice == '4':  # Fast training
            # Ask which algorithm
            algo_choice = input("Which algorithm? (1=Q-Learning, 2=SARSA): ").strip()
            if algo_choice == '1':
                agent, rewards, steps = train_qlearning_agent(episodes=episodes, visualize=False)
            elif algo_choice == '2':
                agent, rewards, steps = train_sarsa_agent(episodes=episodes, visualize=False)
            else:
                print("Invalid choice. Skipping.")
                continue

            print('\nTraining completed!')
            print(f'Average reward (last 100): {np.mean(rewards[-100:]):.2f}')
            print(f'Average steps (last 100): {np.mean(steps[-100:]):.1f}')

        print("\n" + "="*50)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()