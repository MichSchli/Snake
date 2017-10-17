from game.snake_instance import SnakeInstance

snake_instance = SnakeInstance((8,8))

while True:
    snake_instance.pretty_print()
    move = input()
    dist_move = [1 if m == move else 0 for m in ["left", "right", "up", "down"]]
    reward = snake_instance.take_action(dist_move)
    if reward == -1:
        print("Game over.")
        print(snake_instance.score)
        break
    snake_instance.pretty_print()