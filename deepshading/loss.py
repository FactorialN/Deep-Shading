import tensorflow as tf


def SSIMLoss(labels, logits):

    pred = logits
    ssim = tf.math.reduce_mean(tf.image.ssim_multiscale(labels, logits, max_val=5.0))
    loss = (1 - ssim) / 2
    acc = ssim

    return loss, pred, acc


def SquareLoss(labels, logits, z):

    pred = logits
    loss = tf.reduce_mean(tf.square((labels - logits) * z))
    acc = 1 / (loss + 1)

    return loss, pred, acc

def AbsLoss(labels, logits):

    pred = logits
    loss = tf.reduce_sum(tf.abs(labels - logits))
    acc = 1 / (loss + 1)

    return loss, pred, acc
# can be extended