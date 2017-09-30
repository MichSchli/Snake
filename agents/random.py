import tensorflow as tf
from tensorflow.contrib.distributions import Dirichlet


class RandomAgent:

    board_state = None
    score_function = None
    session = None

    def __init__(self):
        self.board_state = tf.placeholder(tf.float32, [None, None])
        self.score_function = self.score_actions(self.board_state)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def compute_action(self, state):
        return self.session.run(self.score_function, feed_dict={self.board_state: state})

    def score_actions(self, state):
        dist = Dirichlet([0.25, 0.25, 0.25, 0.25])
        return dist.sample()