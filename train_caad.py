from __future__ import print_function
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import sys
import pickle
import math
import utils
from model import Generator
from model import Discriminator_nodropout as Discriminator
from utils import weights_init, augment_bt_salt, getDiscriminatorEmbeddings, compute_cosinescore_dynamic, getEmdCentroids, L_supclass, gradient_penalty, augment_bt_rot90
import pdb
import imageio
from sklearn import metrics


args = sys.argv
print('dataset: ', args[1])
# print('model: ', args[2])

dataset = args[1]
# mdl = args[2]

cwd = os.getcwd()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-4
lambda_gp = 10
critic_iterations = 5

outpath = cwd + '/logs/caad/{}'.format(dataset)

if not os.path.isdir(outpath):
    os.makedirs(outpath)

train = np.load(cwd+'/data/{}/train.npy'.format(dataset))

print(train.shape)

with open(cwd+'/data/{}/testwithtarget.pkl'.format(dataset), 'rb') as f:
    read = pickle.load(f)
dat, lab = list(zip(*list(read)))
dat = np.array(dat)
lab = np.array(lab)
print(dat.shape, lab.shape)
test_data = []
for i in range(len(dat)):
    test_data.append([dat[i], lab[i]])

val = np.load(cwd+'/data/{}/val.npy'.format(dataset))


dataloader_train = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True)
dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False)
dataloader_val = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                             shuffle=False)


netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))


G_losses = []
D_losses = []
iters = 0
res_epoch = 0
num_epochs = 100
critic_real_val = []
critic_fake_val = []
reconstructionerror = []

print("Starting Training...")
for epoch in range(res_epoch, num_epochs):

    fake_lastitem = None

    for i, data in enumerate(dataloader_train):

        X = Variable(data[:, 0, :].unsqueeze(1).float()).to(device)
        bsize = X.size(0)
        if dataset != 'mnist':
            anom1 = augment_bt_salt(X)
        else:
            anom1 = augment_bt_rot90(X)

        # Update discriminator
        for idx, _ in enumerate(range(critic_iterations)):

            noise = torch.randn(X.size(0), 1, 10, 10, device=device)
            fake = netG(noise, X)

            critic_real, proj_real = netD(X, penul=True)
            critic_fake, proj_fake = netD(fake, penul=True)
            critic_anom1, proj_anom1 = netD(anom1, penul=True)

            critic_real = critic_real.reshape(-1)
            critic_fake = critic_fake.reshape(-1)
            critic_anom1 = critic_anom1.reshape(-1)

            proj_real = proj_real.reshape(bsize, -1)
            proj_fake = proj_fake.reshape(bsize, -1)
            proj_anom1 = proj_anom1.reshape(bsize, -1)

            L_cont_neg = L_supclass(proj_real, proj_anom1)
            L_cont_pos = L_supclass(proj_anom1, proj_real)

            gp = gradient_penalty(netD, X, fake)
            L_disc = (-(torch.mean(critic_real) -
                        torch.mean(critic_fake))) + lambda_gp*gp
            lossD = L_disc + L_cont_neg + L_cont_pos
            critic_real_val = torch.mean(critic_real).item()
            critic_fake_val = torch.mean(critic_fake).item()
            netD.zero_grad()
            lossD.backward(retain_graph=True)
            optimizerD.step()

        # Update Generator
        output = netD(fake)
        output = output.reshape(-1)
        lossG = -torch.mean(output)
        netG.zero_grad()
        lossG.backward()
        optimizerG.step()

    print('[%d/%d] - Loss G: %.4f, Loss D: %.4f, Loss disc: %.4f, critic(real):%.4f, critic(fake):%.4f' %
          (epoch+1, num_epochs, lossG.item(), lossD.item(), L_disc.item(), critic_real_val, critic_fake_val))

    # Save Losses for plotting later
    G_losses.append(lossG.item())
    D_losses.append(lossD.item())

    if epoch % 10 == 0:
        torch.save(netG.state_dict(), outpath + '/g.pth')
        torch.save(netD.state_dict(), outpath + '/d.pth')
        imageio.imwrite(outpath + '/input.jpeg',
                        X.cpu().data.numpy()[0, 0].astype(np.uint8))
        imageio.imwrite(outpath + '/fake.jpeg',
                        fake.cpu().data.numpy()[0, 0].astype(np.uint8))

torch.save(netG.state_dict(), outpath + '/g.pth')
torch.save(netD.state_dict(), outpath + '/d.pth')


netD.eval()

train_embds = getDiscriminatorEmbeddings(dataloader_train, netD, device)
embds_centroids = getEmdCentroids(train_embds, 1)

val_scores = []
val_dxs = []

for i, data in tqdm(enumerate(dataloader_val)):

    dataBatchX = Variable(data[:, 0, :].unsqueeze(1).float()).to(device)
    b_size = dataBatchX.size(0)
    label_real = torch.full((b_size,), 0, dtype=torch.float)

    df_x, d_x = netD(dataBatchX, penul=True)

    score = compute_cosinescore_dynamic(netD, F.normalize(
        d_x, dim=1).cpu().data.numpy(), device, embds_centroids)
    val_scores.append(score)

val_scores = np.sort(np.concatenate(val_scores))[::-1]
if dataset != 'mnist':
    threshold = val_scores[math.floor(len(val_scores)*0.01)]
else:
    threshold = val_scores[math.floor(len(val_scores)*0.01)]

scores = []
labels = []
d_xs = []
d_gxs = []
testdata = []

for data, _labels in tqdm(dataloader_test):

    # no anomalies
    X = Variable(data.float()).to(device)
    b_size = X.size(0)
    Y = torch.tensor(_labels, dtype=torch.float)

    for data_ in [(X, Y)]:

        _, d_x = netD(data_[0], penul=True)

        score = compute_cosinescore_dynamic(netD, F.normalize(
            d_x, dim=1).cpu().data.numpy(), device, embds_centroids)
        scores.append(score)
        d_xs.append(d_x.reshape(b_size, -1).cpu().data.numpy())
        labels.append(data_[1].squeeze().cpu().data.numpy())
        testdata.append(data_[0][:, 0, :].cpu().data.numpy())

scores = np.concatenate(scores)
labels = np.concatenate(labels)
testdata = np.concatenate(testdata)
d_xs = np.concatenate(d_xs)

testdata_scores_labels = [(testdata[i], scores[i], labels[i])
                          for i in range(len(scores))]
data_noanom = [x[0] for x in testdata_scores_labels if (x[2] == 0)]
data_anom1 = [x[0] for x in testdata_scores_labels if (x[2] == 1)]
scores_noanom = [x[1] for x in testdata_scores_labels if (x[2] == 0)]
scores_anom1 = [x[1] for x in testdata_scores_labels if (x[2] == 1)]

predictions = np.array([1 if s > threshold else 0 for s in scores])

print('****')
print(classification_report(np.concatenate([labels]), np.concatenate(
    [predictions]), target_names=['benign', 'anomaly']))
fpr, tpr, _ = metrics.roc_curve(
    np.concatenate([labels]),  np.concatenate([scores]))
auc_roc = metrics.auc(fpr, tpr)
precision, recall, thresholds = metrics.precision_recall_curve(
    np.concatenate([labels]),  np.concatenate([scores]))
auc_prc = metrics.auc(recall, precision)

print('AUCROC:{}, AUCPRC:{}'.format(auc_roc, auc_prc))
