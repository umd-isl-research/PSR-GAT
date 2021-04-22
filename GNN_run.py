import numpy as np
import pandas as pd
import copy
import json
import os
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv
from torch_geometric.data import DataLoader
from model_build import save_model, PSR_GAT
from data_processing import SYNCScrSessionDataset_v4
from sklearn.metrics import roc_auc_score
from util import user_in3months, noise_user_batch1, noise_user_batch2

def train(sta=None):
    model.train()
    loss_all = 0
    if len(train_dataset) == 0:
        return 1e3
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if sta is None:
            output = model(data)
        else:
            output = model(data, sta)

        label = data.y.to(device)
        loss = crit(output, label.long())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

def getTopNhit(label, logit, N=3):
    """
    :param label: numpy array, [bs]
    :param logit: numpy array, [bs, num_items]
    :param N: topN
    :return: number of successful hits
    """
    assert label.shape[0]==logit.shape[0]  # same number of samples
    indices = logit.argsort(axis=1)[:,-N:]
    return sum([i in j for i, j in zip(label, indices)])

def getTopNRecipocalRank(label, logit, N=3):
    assert label.shape[0] == logit.shape[0]  # same number of samples
    indices = logit.argsort(axis=1)[:, -N:]
    indices = indices[:,::-1]  #reverse, largst prob to smallest prob
    _, rank = np.where(indices == label[:, np.newaxis])
    return np.sum(np.reciprocal(rank+1, dtype=np.float))

def evaluate(loader, sta=None):
    model.eval()
    predictions = []
    labels = []
    acc, topNhit=0, 0
    reciprocal_rank = 0
    if len(loader.dataset) == 0:
        return 0.0, 0.0, 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if sta is None:
                logit = model(data).detach()
                # logit = torch.randn(logit.size(), dtype=logit.dtype).to(device) # benchmark of purely random guess
            else:
                logit = model(data, sta).detach()
            pred = F.softmax(logit, 1).max(1)[1]
            # pred = torch.tensor([data.x[data.batch==i][-1] for i in data.batch.unique(sorted=False)], dtype=pred.dtype).to(device) # benchmark of recommending the last item of a session
            label = data.y.detach().long()
            acc += pred.eq(label.data).cpu().sum().item()
            logit = logit.cpu().numpy()
            label = label.cpu().numpy()
            topNhit += getTopNhit(label, logit, topN)
            reciprocal_rank += getTopNRecipocalRank(label, logit, topN)
            predictions.append(logit)
            labels.append(label)
    predictions = np.vstack(predictions)
    labels = np.hstack(labels)
    #return roc_auc_score(labels, predictions), acc/labels.shape[0]
    return acc/labels.shape[0], topNhit/labels.shape[0], reciprocal_rank/labels.shape[0]


if __name__ == "__main__":  # this is the top code segment to run and evaluate GNN
    mode = "train"  # "train" "test"
    batch_size = 64
    epoch_max = 80
    from util import device  # determine run on cpu or GPU
    # with open("..\\SYNC_preprocessedData\\batch2_userBased_sta.json", "r") as f:  # do not use this, and you do not need this, data leakage problem
        # users_sta = json.load(f)
		# user_sta = None
    user_sta = None
    with open("..\\SYNC_preprocessedData\\Common_90_Users_batch1and2.json", "r") as f: # load user hashed strings you want to run the experiment, or all users, preprocessed in you computer
        users = json.load(f)
        users = set(users)-set(noise_user_batch2)
    performance_HR, performance_MRR = defaultdict(list), defaultdict(list)
    num_sessions = []  # use it if you want to know how many sessions are extracted from each user
    for tar_uid in tqdm(users):
        dataset = SYNCScrSessionDataset_v4(root="SYNC_batch2_BCdivided_multiscale_lastanchor", tar_uid=tar_uid)  # predict the anchor screen using all prior screens, use v6 for early prediction
        data_length = len(dataset)
        num_sessions.append(data_length)

        num_tr = int(data_length * 0.8)
        num_te = data_length - num_tr
        train_dataset = dataset[:num_tr]
        test_dataset = dataset[num_tr:]

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        # tar_sta = users_sta[tar_uid] # do not use tar_sta, data leakage problem
        # tar_freq = tar_sta[0]  # tar_dur = tar_sta[1]
        # tar_freq = [1 if i > 0 else 0 for i in tar_freq]
        # tar_freq = torch.FloatTensor(tar_freq).to(device)

        model = PSR_GAT(num_items=231, head=3).to(device) #239 in batch1, 231 in batch2, without counting those default screens
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1, last_epoch=-1)
        crit = torch.nn.CrossEntropyLoss()
        if mode == "train":
            topN = 3
            best_model, best_hr = None, -1
            for epoch in range(epoch_max):
                loss = train(sta=None)  # sta=tar_freq, remember to set tar_freq to None, or ignore it
                train_acc, train_hitrate, _ = evaluate(train_loader, sta=None)  # sta=tar_freq
                test_acc, test_hitrate, _ = evaluate(test_loader, sta=None)  # sta=tar_freq
                scheduler.step()
                if test_hitrate>best_hr:
                    best_hr = test_hitrate
                    best_model = copy.deepcopy(model)
                print('Epoch: {:03d}/{}, Loss={:.5f}, Train Acc={:.5f}, top{:d} Train hr={:.5f}, Test Acc={:.5f}, Test hr={:.5f}'.
                      format(epoch, epoch_max, loss, train_acc, topN, train_hitrate, test_acc, test_hitrate))
            save_model(best_model, "models/GNN_model_"+tar_uid+".pt")  # save trained models in your drive
        elif mode == "test":
            model.load_state_dict(torch.load("models/MyPrefGATConvH=5e2eGRUs2s_multigbsta_noRepInSess/GNN_model_" + tar_uid + ".pt"))  # load the model you saved in training process
            for topN in (1, 3, 6, 9, 12, 15):
                test_acc, test_hitrate, test_MRR = evaluate(test_loader, sta=None)
                print("Test acc={:.5f}, top{:d} hr={:.5f}, MRR={:.5f}".format(test_acc, topN, test_hitrate, test_MRR))
                performance_HR[tar_uid].append(test_hitrate)
                performance_MRR[tar_uid].append(test_MRR)
    if mode == "test": # save testing results in your drive
        df_HR = pd.DataFrame(performance_HR).transpose()
        df_MRR = pd.DataFrame(performance_MRR).transpose()
        df_HR.to_csv("myGNN HRdiv start5th lastanchor HR.csv", index=True)
        df_MRR.to_csv("myGNN HRdiv start5th lastanchor MRR.csv", index=True)