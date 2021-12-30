import tensorflow as tf
from class_SEPs_and_Capacities_v3 import SEPs


class NeuralNetwork_softplus:
    def __init__(self, name, epsilon, n_antennas, snrdB,
                 size_batch, size_out, learning_rate, n_hidden):
        self.epsilon = epsilon
        self.n_antennas = n_antennas
        self.snrdB = snrdB
        self.size_batch = size_batch
        self.size_out = size_out
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden

        self.hB_row = tf.placeholder(tf.complex64, [None, self.n_antennas])
        self.hE_row = tf.placeholder(tf.complex64, [None, self.n_antennas])
        self.nB = tf.placeholder(tf.complex64, [None, 1])
        self.z_row = tf.placeholder(tf.complex64, [None, n_antennas-1])

        # A LIST of Beamforming Vectors and a LIST of Artificial Noise Matrices
        self.LIST_beam_col = tf.placeholder(tf.complex64, [self.size_batch, self.n_antennas, 1])
        self.LIST_AN = tf.placeholder(tf.complex64, [self.size_batch, self.n_antennas, self.n_antennas-1])

        # some related variables
        self.hET_b = tf.placeholder(tf.complex64, [None, 1])
        self.hET_A_z = tf.placeholder(tf.complex64, [None, 1])
        self.squaredNorm_hET_A = tf.placeholder(tf.float32, [None, 1])
        self.squaredNorm_hET_A_z = tf.placeholder(tf.float32, [None, 1])
        self.squaredNorm_nB = tf.placeholder(tf.float32, [None, 1])
        self.Norm_hB = tf.placeholder(tf.float32, [None, 1])

        self.input, self.output = self.architecture(name)
        ''' Create an object that computes some important expressions '''
        object_SEPs = SEPs(self.epsilon,
                           self.n_antennas,
                           self.snrdB,
                           self.size_batch,
                           self.hB_row,
                           self.hE_row,
                           self.nB,
                           self.z_row,
                           self.LIST_beam_col,
                           self.LIST_AN,
                           self.hET_b,
                           self.hET_A_z,
                           self.squaredNorm_hET_A,
                           self.squaredNorm_hET_A_z,
                           self.squaredNorm_nB,
                           self.Norm_hB,
                           self.output)
        self.objective_function = object_SEPs.objective_function
        self.Cs = object_SEPs.Cs
        self.c_pos = object_SEPs.c_pos
        self.CB = object_SEPs.C_B
        self.CE = object_SEPs.C_E
        self.CSEP_B = object_SEPs.CSEP_B
        self.CSEP_E = object_SEPs.CSEP_E
        #
        self.lambDa_1 = 0.8
        self.lambDa_2 = 1 - self.lambDa_1
        self.delta = 1
        ''' Cost function corresponding to ONLY ONE example '''
        self.cost_of_a_single_example = tf.add(self.lambDa_1*(self.objective_function),
                                               self.lambDa_2*tf.tanh(self.c_pos/self.delta))

        ''' Cost function for training is the average cost over a batch of examples '''
        self.cost_of_a_batch = tf.reduce_mean(self.cost_of_a_single_example)

        ''' Optimizer and training operation '''
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.cost_of_a_batch)

    def architecture(self, name):
        net_in = tf.concat([tf.real(self.hB_row), tf.imag(self.hB_row),
                            tf.real(self.hE_row), tf.imag(self.hE_row)],
                           axis=1)
        net = tf.layers.dense(net_in, self.n_hidden, activation=tf.nn.relu, use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.dense(net, self.n_hidden, activation=tf.nn.relu, use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.contrib.layers.xavier_initializer())
        # net = tf.layers.dense(net, self.n_hidden, activation=tf.nn.relu, use_bias=True,
        #               kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #               bias_initializer=tf.contrib.layers.xavier_initializer())
        # net = tf.layers.dense(net, self.n_hidden, activation=tf.nn.relu, use_bias=True,
        #               kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                bias_initializer=tf.contrib.layers.xavier_initializer())
        net_out = tf.layers.dense(net, self.size_out, activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(),
                                  bias_initializer=tf.constant_initializer(1))
        return net_in, net_out

    def train(self, hB_row_val, hE_row_val, nB_val, z_row_val,
              LIST_beam_col_val, LIST_AN_val,
              hET_b_val, hET_A_z_val, squaredNorm_hET_A_val, squaredNorm_hET_A_z_val,
              squaredNorm_nB_val, Norm_hB_val,
              sess):
        return sess.run(self.train_op,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })

    def get_output(self, hB_row_val, hE_row_val, nB_val, z_row_val,
                   LIST_beam_col_val, LIST_AN_val,
                   hET_b_val, hET_A_z_val,
                   squaredNorm_hET_A_val,
                   squaredNorm_hET_A_z_val,
                   squaredNorm_nB_val,
                   Norm_hB_val,
                   sess):
        return sess.run(self.output,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })

    def get_cost(self, hB_row_val,
                 hE_row_val,
                 nB_val, z_row_val,
                 LIST_beam_col_val,
                 LIST_AN_val,
                 hET_b_val, hET_A_z_val,
                 squaredNorm_hET_A_val,
                 squaredNorm_hET_A_z_val,
                 squaredNorm_nB_val,
                 Norm_hB_val,
                 sess):
        return sess.run(self.cost_of_a_single_example,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })

    def get_CB(self, hB_row_val,
               hE_row_val,
               nB_val,
               z_row_val,
               LIST_beam_col_val,
               LIST_AN_val,
               hET_b_val,
               hET_A_z_val,
               squaredNorm_hET_A_val,
               squaredNorm_hET_A_z_val,
               squaredNorm_nB_val,
               Norm_hB_val,
               sess):
        return sess.run(self.CB,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })

    def get_CE(self, hB_row_val,
               hE_row_val,
               nB_val, z_row_val,
               LIST_beam_col_val,
               LIST_AN_val,
               hET_b_val,
               hET_A_z_val,
               squaredNorm_hET_A_val,
               squaredNorm_hET_A_z_val,
               squaredNorm_nB_val, Norm_hB_val,
               sess):
        return sess.run(self.CE,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })

    def get_CSEP_B(self, hB_row_val,
                   hE_row_val,
                   nB_val,
                   z_row_val,
                   LIST_beam_col_val,
                   LIST_AN_val,
                   hET_b_val, hET_A_z_val,
                   squaredNorm_hET_A_val,
                   squaredNorm_hET_A_z_val,
                   squaredNorm_nB_val,
                   Norm_hB_val,
                   sess):
        return sess.run(self.CSEP_B,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })

    def get_CSEP_E(self, hB_row_val,
                   hE_row_val,
                   nB_val,
                   z_row_val,
                   LIST_beam_col_val,
                   LIST_AN_val,
                   hET_b_val,
                   hET_A_z_val,
                   squaredNorm_hET_A_val,
                   squaredNorm_hET_A_z_val,
                   squaredNorm_nB_val, Norm_hB_val,
                   sess):
        return sess.run(self.CSEP_E,
                        feed_dict={self.hB_row: hB_row_val,
                                   self.hE_row: hE_row_val,
                                   self.nB: nB_val,
                                   self.z_row: z_row_val,
                                   self.LIST_beam_col: LIST_beam_col_val,
                                   self.LIST_AN: LIST_AN_val,
                                   self.hET_b: hET_b_val,
                                   self.hET_A_z: hET_A_z_val,
                                   self.squaredNorm_hET_A: squaredNorm_hET_A_val,
                                   self.squaredNorm_hET_A_z: squaredNorm_hET_A_z_val,
                                   self.squaredNorm_nB: squaredNorm_nB_val,
                                   self.Norm_hB: Norm_hB_val
                                   })
