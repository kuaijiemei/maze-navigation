# 导入所需的库
import numpy as np
import random
import time
import pygame

# 数据准备
class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.end = (size-1, size-1)
        # 初始化障碍物
        for i in range(1, size-1):
            for j in range(1, size-1):
                if random.random() < 0.2:  # 20% 概率生成障碍物
                    self.grid[i][j] = -1

    def reset(self):
        return self.start

    def step(self, state, action):
        x, y = state
        if action == 0:  # 上
            x = max(0, x-1)
        elif action == 1:  # 下
            x = min(self.size-1, x+1)
        elif action == 2:  # 左
            y = max(0, y-1)
        elif action == 3:  # 右
            y = min(self.size-1, y+1)

        next_state = (x, y)
        reward = -1  # 每步惩罚
        done = (next_state == self.end)

        if self.grid[x][y] == -1:  # 碰到障碍物
            reward = -10
            next_state = state  # 碰到障碍物无法移动

        return next_state, reward, done

# 模型构建 - Q-Learning
class QLearning:
    def __init__(self, grid_world, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.grid_world = grid_world
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((grid_world.size, grid_world.size, 4))  # 4 个动作

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state[0]][state[1]])

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state[0]][state[1]][action]
        next_max = np.max(self.q_table[next_state[0]][next_state[1]])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state[0]][state[1]][action] = new_value

# 训练过程
def train(q_learning, episodes=1000):
    rewards = []
    for episode in range(episodes):
        state = grid_world.reset()
        total_reward = 0
        done = False
        while not done:
            action = q_learning.choose_action(state)
            next_state, reward, done = grid_world.step(state, action)
            q_learning.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# 结果评估
def evaluate(q_learning):
    state = grid_world.reset()
    path = [state]
    done = False
    while not done:
        action = np.argmax(q_learning.q_table[state[0]][state[1]])
        state, _, done = grid_world.step(state, action)
        path.append(state)
    return path

# 绘制迷宫
def draw_maze(grid_world, path=None):
    pygame.init()
    size = grid_world.size
    cell_size = 60
    width, height = size * cell_size, size * cell_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Maze Navigation')

    colors = {
        'empty': (255, 255, 255),
        'start': (0, 255, 0),
        'end': (255, 0, 0),
        'obstacle': (0, 0, 0),
        'path': (0, 0, 255),
        'current': (255, 255, 0)  # 当前位置颜色
    }

    running = True
    if path:
        for i, state in enumerate(path):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(colors['empty'])

            # 绘制障碍物
            for x in range(size):
                for y in range(size):
                    if grid_world.grid[x][y] == -1:
                        pygame.draw.rect(screen, colors['obstacle'], (y * cell_size, x * cell_size, cell_size, cell_size))

            # 绘制起点和终点
            pygame.draw.rect(screen, colors['start'], (grid_world.start[1] * cell_size, grid_world.start[0] * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, colors['end'], (grid_world.end[1] * cell_size, grid_world.end[0] * cell_size, cell_size, cell_size))

            # 绘制已走过的路径
            for j in range(i+1):
                if path[j] != grid_world.start and path[j] != grid_world.end:
                    if j == i:  # 当前位置
                        pygame.draw.rect(screen, colors['current'], (path[j][1] * cell_size, path[j][0] * cell_size, cell_size, cell_size))
                    else:
                        pygame.draw.rect(screen, colors['path'], (path[j][1] * cell_size, path[j][0] * cell_size, cell_size, cell_size))

            pygame.display.flip()
            time.sleep(0.5)

            if not running:
                break

    pygame.quit()

# 主程序
if __name__ == "__main__":
    grid_world = GridWorld()
    q_learning = QLearning(grid_world)
    rewards = train(q_learning)
    path = evaluate(q_learning)
    draw_maze(grid_world, path)
    print("路径：", path)
    print("Q 表：\n", q_learning.q_table)