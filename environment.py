import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SimpleMaze:
    """Simple 5x5 maze environment for Q-learning"""

    def __init__(self):
        # 5x5 maze layout
        # 0: empty, 1: trap, 2: target, 3: start
        self.maze = np.array([
            [3, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 2, 0]
        ])

        self.size = 5
        self.start_pos = (0, 0)
        self.target_pos = (4, 3)
        self.traps = [(1, 1), (2, 1), (3, 3), (4, 1)]

        self.reset()

    def reset(self):
        """Reset agent to start position"""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        row, col = self.agent_pos

        if action == 0:  # up
            new_pos = (max(0, row - 1), col)
        elif action == 1:  # right
            new_pos = (row, min(self.size - 1, col + 1))
        elif action == 2:  # down
            new_pos = (min(self.size - 1, row + 1), col)
        elif action == 3:  # left
            new_pos = (row, max(0, col - 1))

        self.agent_pos = new_pos

        # Check if stepped on a trap
        if self.agent_pos in self.traps:
            return self.agent_pos, -50, True  # Large negative reward and terminate

        # Check if reached target
        if self.agent_pos == self.target_pos:
            return self.agent_pos, 100, True

        # Small negative reward for each step
        return self.agent_pos, -1, False

    def get_valid_actions(self, state):
        """Get list of valid actions from current state"""
        row, col = state
        actions = []

        # Check all four directions
        if row > 0:
            actions.append(0)  # up
        if col < self.size - 1:
            actions.append(1)  # right
        if row < self.size - 1:
            actions.append(2)  # down
        if col > 0:
            actions.append(3)  # left

        return actions if actions else [0, 1, 2, 3]  # Return all if none valid


class MazeVisualizer:
    """Real-time maze visualizer using matplotlib"""

    def __init__(self, maze):
        self.maze = maze
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.ion()  # Turn on interactive mode

        # Set up the plot
        self.ax.set_xlim(0, maze.size)
        self.ax.set_ylim(0, maze.size)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(maze.size))
        self.ax.set_yticks(range(maze.size))
        self.ax.set_xticklabels(range(maze.size))
        self.ax.set_yticklabels(range(maze.size))

        # Initialize maze elements
        self.patches = []
        self.agent_circle = None

        self.draw_static_elements()
        plt.show(block=False)

    def draw_static_elements(self):
        """Draw maze traps and target (these don't change)"""
        # Clear previous patches
        for patch in self.patches:
            patch.remove()
        self.patches.clear()

        # Draw maze grid
        for i in range(self.maze.size):
            for j in range(self.maze.size):
                if (i, j) in self.maze.traps:
                    rect = patches.Rectangle((j, self.maze.size - i - 1), 1, 1, linewidth=1,
                                           edgecolor='black', facecolor='red')  # Red for traps
                    self.ax.add_patch(rect)
                    self.patches.append(rect)
                elif (i, j) == self.maze.target_pos:
                    rect = patches.Rectangle((j, self.maze.size - i - 1), 1, 1, linewidth=1,
                                           edgecolor='black', facecolor='yellow')
                    self.ax.add_patch(rect)
                    self.patches.append(rect)
                else:
                    rect = patches.Rectangle((j, self.maze.size - i - 1), 1, 1, linewidth=1,
                                           edgecolor='gray', facecolor='white')
                    self.ax.add_patch(rect)
                    self.patches.append(rect)

        # Add grid lines
        for i in range(self.maze.size + 1):
            self.ax.axhline(y=i, color='gray', linewidth=0.5)
            self.ax.axvline(x=i, color='gray', linewidth=0.5)

    def update_agent(self, agent_pos, episode=None, step=None):
        """Update agent position"""
        # Remove previous agent circle
        if self.agent_circle:
            self.agent_circle.remove()

        # Draw new agent position
        agent_y, agent_x = agent_pos
        self.agent_circle = patches.Circle((agent_x + 0.5, self.maze.size - agent_y - 0.5), 0.3,
                                          linewidth=1, edgecolor='white', facecolor='black')
        self.ax.add_patch(self.agent_circle)

        # Update title (set_title returns None, so we don't need to remove it)
        if episode is not None and step is not None:
            self.ax.set_title(f'Episode: {episode}, Step: {step}')
        else:
            self.ax.set_title('Maze Environment')

        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow GUI update

    def close(self):
        """Close the visualizer"""
        plt.close(self.fig)