import time

import numpy as np

from game.snake_instance import SnakeInstance
from agents.q_learn import QLearnAgent as Agent

snake_instance = SnakeInstance((8,8))
agent = Agent((10,10), 4, 256)

loss_acc = 0
counter = 0
slow = False
e = 0.9
step_counter = 0
score_counter = 0
num_games = 1
for i in range(1000000):
    board_copy = snake_instance.board[:].astype(np.float32)
    move = agent.compute_action(snake_instance.board)
    step_counter += 1

    if np.random.randn(1) < e:
        move = np.random.randint(4)

    dist_move = [1 if m == move else 0 for m in [0,1,2,3]]
    reward = snake_instance.take_action(dist_move)

    if reward == -1:
        loss = agent.compute_update(board_copy, move, snake_instance.board.astype(np.float32), reward, True)
        snake_instance = SnakeInstance((8,8))
        num_games += 1
    else:
        score_counter += reward
        loss = agent.compute_update(board_copy, move, snake_instance.board.astype(np.float32), reward, False)

    if e > 0.1:
        e = e * .9999
    counter += 1
    loss_acc += loss
    if i > agent.batch_size and i % 100 == 0:
        print("Loss at iteration " + str(i) + ": " + str(loss_acc/counter))
        print("Mean game length: " + str(step_counter/num_games))
        print("Mean game score: " + str(score_counter/num_games))
        print("e: " + str(e))

        counter = 0
        loss_acc = 0
        step_counter = 0
        score_counter = 0
        num_games = 1

snake_instance = SnakeInstance((8,8))
while True:
    board_copy = snake_instance.board[:]
    q,move = agent.compute_action(snake_instance.board, get_q=True)
    print(q)
    dist_move = [1 if m == move else 0 for m in [0,1,2,3]]
    reward = snake_instance.take_action(dist_move)
    print(reward)

    if reward == -1:
        print("Game over.")
        print(snake_instance.score)
        snake_instance = SnakeInstance((8,8))

    snake_instance.pretty_print()
    time.sleep(1)
