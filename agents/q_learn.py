import tensorflow as tf
import numpy as np
import time


class QLearnAgent:

    gamma = 0.9
    experiences = None
    experience_pointer = 0
    traversed = False
    batch_size = 64

    def __init__(self, board_shape, embedding_dim, hidden_dim):
        self.experiences = [None]*100000
        self.board_state = tf.placeholder(tf.float32, board_shape)
        self.batch_board_state = tf.placeholder(tf.float32, [None, board_shape[0],  board_shape[1]])
        self.actual_next_q = tf.placeholder(tf.float32, [None,4])

        c1_glorot_var = np.sqrt(6)/np.sqrt(3*3 + hidden_dim)
        c2_glorot_var = np.sqrt(6)/np.sqrt(3*3*hidden_dim + hidden_dim)

        self.C1 = tf.Variable(tf.random_uniform([3,3, 1, hidden_dim], minval=-c1_glorot_var, maxval=c1_glorot_var))
        self.c1_bias = tf.Variable(tf.zeros([hidden_dim]))
        self.C2 = tf.Variable(tf.random_uniform([3,3, hidden_dim, hidden_dim], minval=-c2_glorot_var, maxval=c2_glorot_var))
        self.c2_bias = tf.Variable(tf.zeros([hidden_dim]))

        w1_glorot_var = np.sqrt(6)/np.sqrt(board_shape[0]*board_shape[1]*hidden_dim + hidden_dim)
        self.W1 = tf.Variable(tf.random_uniform([board_shape[0]*board_shape[1]*hidden_dim, hidden_dim], minval=-w1_glorot_var, maxval=w1_glorot_var))
        self.b1 = tf.Variable(tf.zeros([hidden_dim]))

        w2_glorot_var = np.sqrt(6) / np.sqrt(hidden_dim + 4)
        self.W2 = tf.Variable(tf.random_uniform([hidden_dim, 4], minval=-w2_glorot_var, maxval=w2_glorot_var))
        self.b2 = tf.Variable(tf.zeros([4]))

        self.score_function = self.score_actions(tf.expand_dims(self.board_state,0))
        self.batch_score_function = self.score_actions(self.batch_board_state, self.batch_size)
        self.loss_function = self.loss(self.actual_next_q)
        self.update_function = self.update()
        self.action_function = self.take_action()

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def compute_update(self, old_state, move, new_state, reward, dead):
        self.experiences[self.experience_pointer] = (old_state, move, new_state, reward, dead)
        self.experience_pointer += 1

        if self.experience_pointer == len(self.experiences):
            self.experience_pointer = 0
            self.traversed = True

        if not self.traversed and self.experience_pointer < self.batch_size:
            return 0

        if self.traversed:
            batch = np.random.randint(0, len(self.experiences), self.batch_size)
        else:
            batch = np.random.randint(0, self.experience_pointer, self.batch_size)
        #batch = np.concatenate((batch, [self.experience_pointer-1]))

        batch = [self.experiences[b] for b in batch]

        old_state = np.array([b[0] for b in batch])
        move = np.array([b[1] for b in batch])
        new_state = np.array([b[2] for b in batch])
        reward = np.array([b[3] for b in batch])
        dead = np.array([b[4] for b in batch])

        old_q = self.session.run(self.batch_score_function, feed_dict={self.batch_board_state: old_state})
        new_state_q = self.session.run(self.batch_score_function, feed_dict={self.batch_board_state: new_state})
        max_new_state_q = np.max(new_state_q, axis=1)
        max_new_state_q[dead] = 0

        for i in range(self.batch_size):
            old_q[i, move[i]] = reward[i] + self.gamma * max_new_state_q[i]

        loss,_ = self.session.run([self.loss_function, self.update_function], feed_dict={self.batch_board_state: old_state, self.actual_next_q: old_q})
        return loss

    def compute_action(self, state, get_q=False):
        q, action = self.session.run([self.score_function, self.action_function], feed_dict={self.board_state: state})

        if get_q:
            return q,action
        else:
            return action

    def take_action(self):
        return tf.argmax(self.score_function)

    def score_actions(self, state, batch_size=1):
        state = tf.expand_dims(state, -1)

        l1 = tf.nn.relu(tf.nn.conv2d(state, self.C1, [1,1,1,1], "SAME") + self.c1_bias)
        l2 = tf.nn.relu(tf.nn.conv2d(l1, self.C2, [1,1,1,1], "SAME") + self.c2_bias)

        flat_embedding = tf.reshape(l2, [batch_size, -1])
        hidden = tf.nn.relu(tf.matmul(flat_embedding, self.W1) + self.b1)
        return tf.squeeze(tf.matmul(hidden, self.W2)) + self.b2

    def loss(self, actual_next_q):
        predicted_next_q = self.batch_score_function
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(actual_next_q - predicted_next_q), 1))
        return loss

    def update(self):
        parameters_to_optimize = [self.W1, self.W2, self.C1, self.C2, self.b1, self.b2]
        opt_func = tf.train.RMSPropOptimizer(learning_rate=0.001)
        grad_func = tf.gradients(self.loss_function, parameters_to_optimize)
        return opt_func.minimize(self.loss_function) #.apply_gradients(zip(grad_func, parameters_to_optimize))
