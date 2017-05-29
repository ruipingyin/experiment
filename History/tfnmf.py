import numpy as np
import tensorflow as tf

class TFNMF(object):
    def __init__(self, V, rank, learning_rate=0.01):
        self.V = tf.constant(V, dtype=tf.float32)
        shape = V.shape

        self.rank = rank
        self.lr = learning_rate

        #scale uniform random with sqrt(V.mean() / rank)
        scale = 2 * np.sqrt(V.mean() / rank)
        initializer = tf.random_uniform_initializer(maxval=scale)

        self.H =  tf.get_variable("H", [rank, shape[1]],
                                     initializer=initializer)
        self.W =  tf.get_variable(name="W", shape=[shape[0], rank],
                                     initializer=initializer)

        self._build_mu_algorithm()


    def _build_mu_algorithm(self):
        V, H, W = self.V, self.H, self.W
        rank = self.rank
        shape = V.get_shape()

        graph = tf.get_default_graph()

        #save W for calculating delta with the updated W
        W_old = tf.get_variable(name="W_old", shape=[shape[0], rank])
        save_W = W_old.assign(W)

        #Multiplicative updates
        with graph.control_dependencies([save_W]):
            #update operation for H
            Wt = tf.transpose(W)
            WV = tf.matmul(Wt, V)
            WWH = tf.matmul(Wt, tf.matmul(W, H))
            WV_WWH = WV / WWH
            #select op should be executed in CPU not in GPU
            with tf.device('/cpu:0'):
                #convert nan to zero
                WV_WWH = tf.where(tf.is_nan(WV_WWH),
                                    tf.zeros_like(WV_WWH),
                                    WV_WWH)
            H_new = H * WV_WWH
            update_H = H.assign(H_new)

        with graph.control_dependencies([save_W, update_H]):
            #update operation for W (after updating H)
            Ht = tf.transpose(H)
            VH = tf.matmul(V, Ht)
            WHH = tf.matmul(tf.matmul(W, H), Ht)
            VH_WHH = VH / WHH
            with tf.device('/cpu:0'):
                VH_WHH = tf.where(tf.is_nan(VH_WHH),
                                        tf.zeros_like(VH_WHH),
                                        VH_WHH)
            W_new = W * VH_WHH
            update_W = W.assign(W_new)

        self.delta = tf.reduce_sum(tf.abs(W_old - W))

        self.step = tf.group(save_W, update_H, update_W)


    def run(self, sess, max_iter=200, min_delta=0.001):

        tf.global_variables_initializer().run()

        for i in xrange(max_iter):
            self.step.run()
            delta = self.delta.eval()
            if delta < min_delta:
                break
        W = self.W.eval()
        H = self.H.eval()
        return W, H
        
