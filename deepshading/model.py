import tensorflow as tf
import structures
import loss

class Model:

    def __init__(self, n, channel, learning_rate=0.1, learning_rate_decay=0.995):

        self.x = tf.placeholder(tf.float32, [None, n, n, channel])
        self.o = tf.placeholder(tf.float32, [None, n, n, 1])
        self.z = tf.placeholder(tf.float32, [None, n, n, 1])

        self.loss, self.pred,  self.acc = self.forward(tf.AUTO_REUSE)

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        """for item in self.params:
            print(item)"""

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, reuse):
        x = self.x
        o = self.o
        z = self.z

        logits = structures.AutoEncoderLikeCNN(x, reuse)

        return loss.SquareLoss(o, logits, z)
