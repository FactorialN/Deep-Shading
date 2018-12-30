import torch
import pytorch_ssim


def SSIMLoss(labels, logits):

    pred = logits
    ssim=pytorch_ssim.ssim(labels,logits)
    #ssim = tf.math.reduce_mean(tf.image.ssim_multiscale(labels, logits, max_val=5.0))
    loss = (1 - ssim) / 2
    acc = ssim

    return loss, pred, acc


def SquareLoss(labels, logits):

    pred = logits
    loss = torch.mean(torch.pow(labels - logits,2))
    acc = 1 / (loss + 1)

    return loss, pred, acc


def AbsLoss(labels, logits):

    pred = logits
    loss = torch.mean(torch.abs(labels - logits))
    acc = 1 / (loss + 1)

    return loss, pred, acc

def CrossEntropyLoss(labels, logits):

    pred = logits
    loss = torch.mean(torch.exp(torch.abs(labels - logits)))
    acc = 1 / (loss + 1)

    return loss, pred, acc

# can be extended