from read import Input, OtherInput
from model import Model
import tensorflow as tf
import numpy as np
import os
import time




def train(model, sess):
    loss, acc = 0.0, 0.0
    times = 0
    for i in range(iter_per_epoch):
        x, y, z = data.sample(batch_size)
        feed = {model.x: x, model.o: y, model.z: z}
        loss_, acc_, _ = sess.run([model.loss, model.acc, model.train_op], feed)
        loss += loss_
        acc += acc_
        times += 1
    loss /= times
    acc /= times
    return acc, loss

def test(model, sess):
    loss, acc = 0.0, 0.0
    times = 0
    x = data.test
    y = data.output
    z = data.outref
    feed = {model.x: x, model.o: y, model.z: z}
    loss_, acc_, output_ = sess.run([model.loss, model.acc, model.pred], feed)
    loss += loss_
    acc += acc_
    times += 1
    loss /= times
    acc /= times
    return acc, loss, output_

data = OtherInput()
train_dir = data.command['train_dir']
learning_rate = data.command['learning_rate']
learning_rate_decay = data.command['learning_rate_decay']
epochs = data.command['epochs']
batch_size = data.command['batch_size']
iter_per_epoch = data.command['iter_per_epoch']
is_training = data.command['is_training']
n = data.data.shape[1]
channel = data.data.shape[3]

with tf.Session() as sess:
    if is_training:
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        model = Model(n, channel, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
        if tf.train.get_checkpoint_state(train_dir):
            model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            tf.global_variables_initializer().run()

        pre_losses = [1e18] * 3
        best_val_acc = 0.0

        for epoch in range(epochs):
            start_time = time.time()
            train_acc, train_loss = train(model, sess)

            model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=model.global_step)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(model.learning_rate.eval()))
            print("  training loss:                 " + str(train_loss))
            print("  training accuracy:             " + str(train_acc))

            if train_loss > max(pre_losses):
                sess.run(model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [train_loss]
    else:
        model = Model(n, channel, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
        if tf.train.get_checkpoint_state(train_dir):
            model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            tf.global_variables_initializer().run()
    
        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        start_time = time.time()
        test_acc, test_loss, test_out = test(model, sess)

        epoch_time = time.time() - start_time
        print("  test loss:                 " + str(test_loss))
        print("  test accuracy:             " + str(test_acc))

        #test_out /= 1000
        f = open('output.txt', 'w')
        test_out = test_out #* data.outref / 100
        #print(np.sum(np.abs((test_out-data.output) * data.output)))
        for i in range(0, test_out.shape[1]):
            for j in range(0, test_out.shape[2]):
                f.write((' ').join(list(map(str, data.outref[0][i][j]))) + ' ' + (' ').join(list(map(str, test_out[0][i][j]))) +' ' + (' ').join(list(map(str, data.output[0][i][j]))) + '\n')
        f.close()

"""
What to show at last

The Conduction
The Model
The realization of TensorFlow and PyTorch
The result of the first method
The disadvantage of the first method
The theorem of PRT
The method of training transfer function
The trivial result of this method
The shortcoming of this method
"""