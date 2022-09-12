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
from model import Generator, Discriminator
from utils import weights_init, augment_bt_salt, getDiscriminatorEmbeddings, compute_cosinescore_dynamic, getEmdCentroids, L_supclass, gradient_penalty
import pdb
import imageio
from sklearn import metrics


args = sys.argv
print('dataset: ', args[1])
# print('model: ', args[2])

dataset = args[1]
assert dataset == 'ltw1' or dataset == 'stw1', "Not valid datasets."
# mdl = args[2]

cwd = os.getcwd()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-4
lambda_gp = 10
critic_iterations = 5

outpath = cwd + '/logs/caad-uq/'

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
        anom1 = augment_bt_salt(X)

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
netD.main[2].train()
netD.main[6].train()
netD.main[10].train()


train_embds = getDiscriminatorEmbeddings(dataloader_train, netD, device)
embds_centroids = getEmdCentroids(train_embds, 1)

val_scores = []
val_dxs = []

for i, data in tqdm(enumerate(dataloader_val)):

    dataBatchX = Variable(data[:, 0, :].unsqueeze(1).float()).to(device)

    df_x, d_x = netD(dataBatchX, penul=True)

    scores_drp = []

    for _ in range(20):
        scores_drp.append(compute_cosinescore_dynamic(netD, F.normalize(
            d_x, dim=1).cpu().data.numpy(), device, embds_centroids))

    val_scores.append(np.array(scores_drp).T.mean(axis=1))

val_scores = np.sort(np.concatenate(val_scores))[::-1]
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

        scores_drp = []
        for _ in range(20):
            _, d_x = netD(data_[0], penul=True)
            scores_drp.append(compute_cosinescore_dynamic(netD, F.normalize(
                d_x, dim=1).cpu().data.numpy(), device, embds_centroids))

        scores.append(np.array(scores_drp))
        labels.append(data_[1].squeeze().cpu().data.numpy())
scores = [x.T for x in scores]
scores = np.concatenate(scores)
labels = np.concatenate(labels)
predictions = np.array(
    [1 if s > threshold else 0 for s in scores.mean(axis=1)])

print('****')
print(classification_report(np.concatenate([labels]), np.concatenate(
    [predictions]), target_names=['benign', 'anomaly']))
fpr, tpr, _ = metrics.roc_curve(np.concatenate(
    [labels]),  np.concatenate([scores.mean(axis=1)]))
auc_roc = metrics.auc(fpr, tpr)
precision, recall, thresholds = metrics.precision_recall_curve(
    np.concatenate([labels]),  np.concatenate([scores.mean(axis=1)]))
auc_prc = metrics.auc(recall, precision)
print('AUCROC:{}, AUCPRC:{}'.format(auc_roc, auc_prc))

preds = []
for scores_20 in scores:
    preds.append(np.array([1 if s > threshold else 0 for s in scores_20])/20)
cmap = [sum(x) for x in preds]

preds_all = []
for scores_20 in scores:
    preds_all.append(np.array([1 if s > threshold else 0 for s in scores_20]))
preds_all = np.array(preds_all)
print(preds_all.shape)


def getdomclass(inst):
    total1s = inst.sum()
    total0s = 20 - total1s
    if total1s >= total0s:
        return 1
    return 0


newpreds = []

for instance in preds_all:
    domclass = getdomclass(instance)
    if domclass == 1:
        newpreds.append(instance.sum())
    else:
        newpreds.append(20-instance.sum())

df = pd.DataFrame(list(zip(newpreds, lab)), columns=['preds', 'labels'])

hil_indices = list(df.preds.sort_values(ascending=False)
                   [round(len(df)*(1-5/100)):].index)
nohil_indices = list(df.preds.sort_values(ascending=False)
                     [:round(len(df)*(1-5/100))].index)

df_labels_hil = df.iloc[hil_indices].labels
df_labels_nohil = df.iloc[nohil_indices].labels

benign_hil_indices = list(df_labels_hil[df_labels_hil == 0].index)
anomaly_hil_indices = list(df_labels_hil[df_labels_hil == 1].index)
benign_nohil_indices = list(df_labels_nohil[df_labels_nohil == 0].index)
anomaly_nohil_indices = list(df_labels_nohil[df_labels_nohil == 1].index)

np.save(outpath+'/benign_hil.npy', dat[benign_hil_indices])
np.save(outpath+'/anomaly_hil.npy', dat[anomaly_hil_indices])
print(dat[benign_hil_indices].shape, dat[anomaly_hil_indices].shape)

np.save(outpath+'/benign_hil_indices.npy', benign_hil_indices)
np.save(outpath+'/anomaly_hil_indices.npy', anomaly_hil_indices)
len(benign_hil_indices), len(anomaly_hil_indices)
np.save(outpath+'/benign_nohil_indices.npy', benign_nohil_indices)
np.save(outpath+'/anomaly_nohil_indices.npy', anomaly_nohil_indices)
len(benign_nohil_indices), len(anomaly_nohil_indices)
