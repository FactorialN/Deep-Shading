from read import Input, OtherInput
from model import Model
import torch
import os
import time
import numpy as np




def train(model, opti):
    loss, acc = 0.0, 0.0
    times = 0
    for i in range(iter_per_epoch):
        x, y = data.sample(batch_size)
        model.zero_grad()
        loss_, output_, acc_=model(x,y)
        loss_.backward()
        opti.step()

        loss += loss_.item()
        acc += acc_.item()
        times += 1
    loss /= times
    acc /= times
    return acc, loss

def test(model):
    loss, acc = 0.0, 0.0
    times = 0
    x = data.test
    y = data.output
    model.zero_grad()
    loss_, output_, acc_=model(x,y)

    loss += loss_.item()
    acc += acc_.item()
    times += 1
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

if is_training == 1:
    
    model = Model(n, channel, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
    opti=torch.optim.Adam(model.parameters())
    if not os.path.exists(train_dir):
        #os.mkdir(train_dir)
        pass
    else:
        model = torch.load(train_dir)
    

    pre_losses = [1e18] * 3
    best_val_acc = 0.0
    for epoch in range(epochs):
        start_time = time.time()
        train_acc, train_loss = train(model, opti)


        epoch_time = time.time() - start_time
        print("Epoch " + str(epoch + 1) + " of " + str(epochs) + " took " + str(epoch_time) + "s")
        print("  training loss:                 " + str(train_loss))
        print("  training accuracy:             " + str(train_acc))

        if train_loss < pre_losses[len(pre_losses)-1]:
            torch.save(model, train_dir)

        pre_losses = pre_losses[1:] + [train_loss]

else:
    model = Model(n, channel, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
    model = torch.load(train_dir)

    pre_losses = [1e18] * 3
    best_val_acc = 0.0
    start_time = time.time()
    test_acc, test_loss, test_out = test(model)

    epoch_time = time.time() - start_time
    print("  test loss:                 " + str(test_loss))
    print("  test accuracy:             " + str(test_acc))

    f = open('output.txt', 'w')
    test_out = test_out.detach().numpy()
    #print(np.mean(np.abs(test_out-data.output)))
    for i in range(0, test_out.shape[1]):
        for j in range(1, test_out.shape[2]):
            f.write((' ').join(list(map(str, test_out[0][i][j]))) + '\n')
    f.close()