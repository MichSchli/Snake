import numpy as np


class SnakeInstance:

    board = None
    snake_head_location = None
    snake_head_direction = None
    memorized_locations = None
    score = None

    def __init__(self, dim):
        self.board = np.zeros(dim, dtype=np.int32)
        self.initialize_snake()
        self.spawn_pellet()
        self.score = 0

        self.pretty_print()

    def initialize_snake(self):
        buffer = 3
        self.snake_head_location = (np.random.randint(self.board.shape[0]-2*buffer)+buffer,
                                    np.random.randint(self.board.shape[1]-2*buffer)+buffer)
        self.board[self.snake_head_location] = 1

        if self.snake_head_location[0] < self.snake_head_location[1]:
            if self.snake_head_location[0] < self.board.shape[0]/2:
                self.board[(self.snake_head_location[0] - 1, self.snake_head_location[1])] = 1
                self.board[(self.snake_head_location[0] - 2, self.snake_head_location[1])] = 1
                self.snake_head_direction = "down"
                self.memorized_locations = [(self.snake_head_location[0] - 2, self.snake_head_location[1]),
                                            (self.snake_head_location[0] - 1, self.snake_head_location[1]),
                                            self.snake_head_location]
            else:
                self.board[(self.snake_head_location[0] + 1, self.snake_head_location[1])] = 1
                self.board[(self.snake_head_location[0] + 2, self.snake_head_location[1]),
                           self.snake_head_location] = 1
                self.snake_head_direction = "up"
                self.memorized_locations = [(self.snake_head_location[0] + 2, self.snake_head_location[1]),
                                            (self.snake_head_location[0] + 1, self.snake_head_location[1]),
                                            self.snake_head_location]
        else:
            if self.snake_head_location[1] < self.board.shape[1]/2:
                self.board[(self.snake_head_location[0], self.snake_head_location[1] - 1)] = 1
                self.board[(self.snake_head_location[0], self.snake_head_location[1] - 2)] = 1
                self.snake_head_direction = "right"
                self.memorized_locations = [(self.snake_head_location[0], self.snake_head_location[1] - 2),
                                            (self.snake_head_location[0], self.snake_head_location[1] - 1),
                                            self.snake_head_location]
            else:
                self.board[(self.snake_head_location[0], self.snake_head_location[1] + 1)] = 1
                self.board[(self.snake_head_location[0], self.snake_head_location[1] + 2)] = 1
                self.snake_head_direction = "left"
                self.memorized_locations = [(self.snake_head_location[0], self.snake_head_location[1] + 2),
                                            (self.snake_head_location[0], self.snake_head_location[1] + 1),
                                            self.snake_head_location]

    def spawn_pellet(self):
        available = np.array(np.where(self.board == 0)).transpose()
        pellet_location = available[np.random.randint(available.shape[0])]
        self.board[pellet_location[0], pellet_location[1]] = 2

    def get_available_actions(self):
        return [0 if move == self.snake_head_direction else 1 for move in ["left", "right", "up", "down"]]

    def take_action(self, action):
        new_head_location = (self.snake_head_location[0]-action[2]+action[3],
                             self.snake_head_location[1]-action[0]+action[1])

        if self.board[new_head_location] == 1:
            return -1

        self.snake_head_location = new_head_location

        if self.board[new_head_location] == 2:
            self.score += 1
            self.board[new_head_location] = 1
            self.spawn_pellet()
            self.memorized_locations.append(new_head_location)
            return 1
        else:
            self.board[new_head_location] = 1
            tail = self.memorized_locations.pop(0)
            self.board[tail] = 0
            self.memorized_locations.append(new_head_location)
            return 0

    def pretty_print(self):
        print("\n".join([" ".join([str(i) for i in row]) for row in self.board]))