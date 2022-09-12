import os
import sys
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import math
from utils import getDiscriminatorEmbeddings, getEmdCentroids, compute_cosinescore_dynamic
from model import Discriminator, Discriminator_nodropout
from sklearn import metrics

args = sys.argv
print('dataset: ', args[1])
print('model: ', args[2])

dataset = args[1]
mdl = args[2]

cwd = os.getcwd()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-4
lambda_gp = 10
critic_iterations = 5

cwd = os.getcwd()

train = np.load(cwd+'/data/{}/train.npy'.format(dataset))

print(train.shape)

dataloader_train = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True)

with open(cwd+'/data/{}/testwithtarget.pkl'.format(dataset), 'rb') as f:
    read = pickle.load(f)

dat, lab = list(zip(*list(read)))
dat = np.array(dat)
lab = np.array(lab)
print(dat.shape, lab.shape)

test_data = []
for i in range(len(dat)):
    test_data.append([dat[i], lab[i]])

dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False)

val = np.load(cwd+'/data/{}/val.npy'.format(dataset))
dataloader_val = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                             shuffle=False)

if mdl != 'caad':
    netD = Discriminator().to(device)
else:
    netD = Discriminator_nodropout().to(device)
netD.load_state_dict(torch.load(
    cwd+'/trainedmodels/{}/{}.pth'.format(dataset, mdl)))
netD.eval()

if mdl != 'caad':
    netD.main[2].train()
    netD.main[6].train()
    netD.main[10].train()

train_embds = getDiscriminatorEmbeddings(dataloader_train, netD, device)
embds_centroids = getEmdCentroids(train_embds, 1)


if mdl == 'caad':
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
        threshold = val_scores[math.floor(len(val_scores)*0.1)]
else:
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


if mdl == 'caad':
    scores = []
    labels = []

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
            labels.append(data_[1].squeeze().cpu().data.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
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

else:
    scores = []
    labels = []

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
