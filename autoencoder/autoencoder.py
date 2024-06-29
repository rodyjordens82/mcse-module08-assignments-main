import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import timedelta
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':
    # Get and distort the data from images
    plot_blocking = False
    df = pd.read_csv('data/mnist_test.csv')
    anom = df[:1000].copy()
    clean = df[1000:].copy()

    # Add random noise to images
    anom.iloc[:, 1:] = anom.iloc[:, 1:].applymap(lambda x: min(255, x + random.randint(100, 200)))
    anom['label'] = 1
    clean['label'] = 0

    # Concatenate and shuffle data
    data_full = pd.concat([anom, clean]).sample(frac=1).reset_index(drop=True)
    data_full.to_csv('anom.csv', index=False)

    # Autoencoder
    class AE(nn.Module):
        def __init__(self):
            super(AE, self).__init__()
            self.enc = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            self.dec = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.ReLU()
            )
        def forward(self, x):
            encode = self.enc(x)
            decode = self.dec(encode)
            return decode

    # training params
    print(' TRAINING *********')
    batch_size = 32
    lr = 1e-2         # learning rate
    w_d = 1e-5        # weight decay
    momentum = 0.9
    epochs = 15

    train_set = pd.read_csv('data/mnist_train.csv', index_col=False)
    train_ = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=12,
                pin_memory=True,
                drop_last=True
            )

    metrics = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AE()
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)
    model.train()
    start = time.time()

    print(' TRAINING THE EPOCHS ********* ')
    for epoch in range(epochs):
        ep_start = time.time()
        running_loss = 0.0

        try:
            for i in range(len(train_)):
                data = torch.from_numpy(np.array(train_set.iloc[i][1:]) / 255).float()
                sample = model(data.to(device))
                loss = criterion(data.to(device), sample)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(' TRAINING NEXT    EPOCH ********* ')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

        epoch_loss = running_loss/len(train_set)
        metrics['train_loss'].append(epoch_loss)
        ep_end = time.time()
        print('-----------------------------------------------')
        print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1,epochs,epoch_loss))
        print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
    end = time.time()
    print('---------****---------------------------------')
    print('[System Complete: {}]'.format(timedelta(seconds=end-start)))

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.set_title('Loss')
    ax.plot(metrics['train_loss'])
    plt.show(block=False)
    print(' Continue from loss plot ********* ')

    loss_dist = []
    anom = pd.read_csv('anom.csv', index_col=False)
    #for bx, data in enumerate(test_):
    for i in range(len(anom)):
        data = torch.from_numpy(np.array(anom.iloc[i][1:])/255).float()
        sample = model(data.to(device))
        loss = criterion(data.to(device), sample)
        loss_dist.append(loss.item())
    print(' Calculated anom data ********* ')

    # scatter plot loss
    loss_sc = []
    for i in loss_dist:
        loss_sc.append((i,i))
    plt.figure()
    plt.scatter(*zip(*loss_sc))
    plt.axvline(0.3, 0.0, 1)
    lower_threshold = 0.0
    upper_threshold = 0.3
    plt.figure(figsize=(10,6))
    plt.show(block=plot_blocking)

    # histogram plot loss
    plt.title('Loss Distribution')
    sns.histplot(loss_dist,bins=100,kde=True, color='blue')
    plt.axvline(upper_threshold, 0.0, 10, color='r')
    plt.axvline(lower_threshold, 0.0, 10, color='b')
    plt.show(block=plot_blocking)

    df = pd.read_csv('anom.csv', index_col=False)
    ddf = pd.DataFrame(columns=df.columns)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_anom = 0

    print(' Calc false positives ********* ')
    # Calc false positives
    for i in range(len(loss_dist)):
        total_anom += df.iloc[i]['label']
        if loss_dist[i] >= upper_threshold:
            n_df = pd.DataFrame([df.iloc[i]])
            n_df['loss'] = loss_dist[i]
            ddf = pd.concat([df,n_df], sort = True)
            if float(df.iloc[i]['label']) == 1.0:
                tp += 1
            else:
                fp += 1
        else:
            if float(df.iloc[i]['label']) == 1.0:
                fn += 1
            else:
                tn += 1
    print('[TP] {}\t[FP] {}\t[MISSED] {}'.format(tp, fp, total_anom-tp))
    print('[TN] {}\t[FN] {}'.format(tn, fn))

    conf = [[tn,fp],[fn,tp]]
    plt.figure()
    sns.heatmap(conf,annot=True,annot_kws={"size": 16},fmt='g')
    print("Plot heatmap")
    plt.show(block=True)

    print("Done")
