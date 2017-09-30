from game.snake_instance import SnakeInstance
from agents.random import RandomAgent as Agent

snake_instance = SnakeInstance((8,8))
agent = Agent()

first = agent.compute_action(snake_instance.board)
print(first)