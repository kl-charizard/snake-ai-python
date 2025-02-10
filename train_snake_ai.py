#!/usr/bin/env python3
import matplotlib.pyplot as plt
from snake_ai_core import SnakeGameAI, Agent

def train():
    scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # 训练时关闭窗口渲染
    game = SnakeGameAI(render=False)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('游戏局数')
    ax.set_ylabel('得分')

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'局数: {agent.n_games} 得分: {score} 最高分: {record}')
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games

            ax.clear()
            ax.plot(scores, label='得分')
            ax.set_xlabel('游戏局数')
            ax.set_ylabel('得分')
            ax.set_title(f'平均得分: {mean_score:.2f}')
            ax.legend()
            plt.pause(0.001)

if __name__ == '__main__':
    train()
