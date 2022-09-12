import random
import torch
import decimal
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from skimage.util import random_noise
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from enum import Enum
from scipy.spatial import distance
from scipy.ndimage import shift
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
        
def getDiscriminatorEmbeddings(dataloader_train,discriminator,device):
    
    d_xs = []
    for data in dataloader_train:
        X = data.float().to(device)
        b_size = X.size(0)
        _,d_x = discriminator(X,penul=True)
        d_xs.append(d_x.reshape(b_size, -1).cpu().data.numpy())
    d_xs = np.concatenate(d_xs)
    return d_xs

def getEmdCentroids(embds,nclusters=4):
    
    embds_norm = preprocessing.normalize(embds)
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(embds_norm)
    indices_dict = {}
    for i in range(nclusters):
        indices_dict[i] = [a[0] for a in list(np.argwhere(kmeans.labels_==i))]
    train_embds_centroids = []
    for i in range(nclusters):
        train_embds_centroids.append(np.sum(embds[indices_dict[i]],axis=0)/len(indices_dict[i]))      
    return train_embds_centroids

def compute_cosinescore_dynamic(netD,Y,device,train_centroids):
    
    d_traincentroids = torch.tensor(train_centroids).float().to(device)
    sims = []
    for i in range(len(Y)):
        maxdistance = 0 
        for centroid in d_traincentroids.cpu().data.numpy():
            dist = distance.cosine(centroid.reshape(-1),Y[i].reshape(-1))
            if not maxdistance > dist:
                maxdistance = dist
        sims.append(maxdistance*norm(Y[i].reshape(-1)))
    
    return sims

def augment_bt_salt(dataBatchX,amountofnoise = 0.001):
    
    device = dataBatchX.device
    b_size = dataBatchX.size(0)
    anomBatchX = Variable(torch.tensor(random_noise(Variable(torch.zeros(dataBatchX.shape)).float().detach().cpu(), mode='salt', clip=True, amount = amountofnoise))).float().to(device)
    anomBatchX =  Variable(random.uniform(dataBatchX.mean().detach(), dataBatchX.max().detach()) * anomBatchX.detach() + dataBatchX.detach()).to(device)
        
    return anomBatchX

def gradient_penalty(D, images, gen_images):
    
    batch_size = images.size(0)
    _device = images.device
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(images)
    alpha = alpha.to(_device)
    interpolated = alpha * images.data + (1 - alpha) * gen_images.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(_device)
    prob_interpolated = D(interpolated)
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(_device),
                           create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def L_supclass(out1, out2, temperature=0.1, distributed=False):

    N = out1.size(0)
    _out = [F.normalize(out1, dim=1), F.normalize(out2, dim=1)]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[N:,N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[N:]
    mask = mask[N:]
    mask = mask / mask.sum(1, keepdim=True)
    lsm = F.log_softmax(sim_matrix, dim=1)

    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()

    return d_loss

def augment_bt_rot90(dataBatchX):
    
    device = dataBatchX.device
    
    b_size = dataBatchX.size(0)
    
    anomBatchX = torch.rot90(dataBatchX,k=1,dims=[2,3]).to(device)
        
    return anomBatchX