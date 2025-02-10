#!/usr/bin/env python3
import time
import torch
from snake_ai_core import SnakeGameAI, Agent, device

def demo():
    # 演示时开启窗口渲染
    game = SnakeGameAI(render=True)
    agent = Agent()
    # 加载训练好的模型（请确保 model.pth 文件存在）
    agent.model.load_state_dict(torch.load('model.pth', map_location=device))
    agent.model.eval()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        if done:
            print('最终得分:', score)
            game.reset()
            time.sleep(1)

if __name__ == '__main__':
    demo()
