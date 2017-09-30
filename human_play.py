from game.snake_instance import SnakeInstance

snake_instance = SnakeInstance((8,8))

while True:
    move = input()
    dist_move = [1 if m == move else 0 for m in ["left", "right", "up", "down"]]
    snake_instance.take_action(dist_move)
    snake_instance.pretty_print()