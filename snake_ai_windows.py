#!/usr/bin/env python3
import pygame
import random
from enum import Enum
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time

# 设备设置：在 Windows 上优先使用 CUDA，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定义方向枚举类
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 定义点类，便于表示游戏中的坐标
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# 贪吃蛇游戏环境
class SnakeGameAI:
    def __init__(self, w=640, h=480, block_size=20, render=True):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.render = render
        self.reset()
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('AI Snake')
            self.clock = pygame.time.Clock()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - self.block_size, self.head.y),
                      Point(self.head.x - 2 * self.block_size, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        if self.render:
            # 处理 pygame 事件（防止窗口无响应）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        self._move(action)  # 根据动作更新蛇头
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(20)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 撞墙检测
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # 撞到自身
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        # 绘制蛇
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, (0, 128, 0), pygame.Rect(pt.x + 4, pt.y + 4, self.block_size - 8, self.block_size - 8))
        # 绘制食物
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        # 显示分数
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action 格式为 [直行, 右转, 左转]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # 不改变方向
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # 右转
        else:  # [0, 0, 1] 左转
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # 危险（直行）
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # 危险（右转）
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # 危险（左转）
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),

            # 当前方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 食物相对于蛇头的位置
            self.food.x < head.x,  # 食物在左边
            self.food.x > head.x,  # 食物在右边
            self.food.y < head.y,  # 食物在上边
            self.food.y > head.y   # 食物在下边
        ]
        return np.array(state, dtype=int)

# 定义深度 Q 网络
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)  # dropout 防止过拟合
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

# 定义 Q 网络的训练器
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# 定义智能体 Agent
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # 探索率
        self.gamma = 0.9  # 折扣率
        self.memory = deque(maxlen=100_000)
        self.batch_size = 1000
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # 随着游戏局数增加逐渐降低探索率
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

# 训练过程：训练过程中会自动保存模型
def train():
    scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(render=False)

    while True:
        # ... 原有的游戏循环代码 ...

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # 每50局自动保存模型检查点
            if agent.n_games % 50 == 0:
                agent.model.save(f'model_checkpoint_{agent.n_games}.pth')

            if score > record:
                record = score
                agent.model.save()  # 新纪录仍保存为model.pth

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

# 演示过程：加载模型并显示 AI 玩游戏的过程
def demo():
    game = SnakeGameAI(render=True)
    agent = Agent()
    # 加载训练好的模型，注意 map_location 参数确保在当前设备上加载
    agent.model.load_state_dict(torch.load("model.pth", map_location=device))
    agent.model.eval()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        if done:
            print("Final Score:", score)
            game.reset()
            time.sleep(1)

if __name__ == "__main__":
    # 默认进入训练模式；如果运行时传入 "demo" 参数，则进入演示模式
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        train()
