import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_user_batch1 = ["4147dc2fc0de3a8fe6e1340155014ff21405a3cbbeab5c49fabbedf1311daf47",  # this users is in batch1, does not have session data
                     "091ce02876feb48e356de94dd0e5f761f786cfebc3c9f5ac765f711a97bf2e98"]  # in batch 1, all 10 sessions of this user are very short
noise_user_batch2 = []

user_in3months=("e4c7c89abcf87b1d6d203ac5732bb08cb4fa9176f1338db570f7ac7cf93ebdbc",
"a83a22a27c4bd7672a7b59b39afb2e27d27869946598b4c57375dfd7f75f5af5",
"4103b726a4374473715e3c7d72af43b89324756a836170bda506b18d1b42c4ea",
"685657c8476e62ad860444db02f4ee47b76c89e81eb505164ecd39f6a82fb1df",
"bcaf01b71107c9c30188b3d08969d8eae53ca8e2631b7dc978cade64b7d5d4bf",
"d35a54312955e2532aa0f3e032248edf3b23218c098eff69d6d9b8fd2b718275",
"4f5412ba1679d659be91b1f0f840a1f98738ed98d58b97c664e171f9ef4adfbd",
"2a1b95c15576e90878d78b6ce5a729d864a6d057a54ea5f409f1151fc848944a",
"01b57b2940381622469b29fba7b70dfd75f3786e0ee60e6ea0b04c1a9ab9e909",
"63a831134622e4a1197ba16c84064c1822351f682ebf4db09f1190fc9b71af7a",
"3fd505a31ffe97228afdb7be20c3e39eac6b23a90fe9126ff423b3b931602e2d",
"9c08dfe79db56fa0a754de27462c4550ad926e85e3a2b880dc1847c4654c95c9",
"d1cfee413ffc0270aecdf1fad01c8f6856a1e4c0595e34cf3a0613826308e18b")
# user "4147dc2fc0de3a8fe6e1340155014ff21405a3cbbeab5c49fabbedf1311daf47" is not included in all_screens.json, exlucding this one, len(all_screens)=191
alpha=-0.02
def duration2weight(d):
    """
    :param d: list of view durations
    :return:  list of normalized durations as edge weight
    """
    if isinstance(d, list):
        return [np.exp(alpha*i) for i in d]
    elif isinstance(d, torch.Tensor):
        return torch.exp(d*alpha)
    else:
        print("not in consideration, function name: duration2weight")

def weight2duration(w):
    return [np.log(i)/alpha if i >0 else 999 for i in w ]

def norm_torchtensor(t, a=0, b=1): # linearly maps elements in t into range (a,b), if t is 2d, apply row-wise
    if t.dim()==1:
        mint,maxt=t.min(), t.max()
        if mint!=maxt:
            return (t-mint)/(maxt-mint)*(b-a)+a
        else:
            # print("all element are the same, cannot use minmax normalization")
            return t # torch.zeros_like(t)
    elif t.dim()==2:
        mint, _ = torch.min(t,dim=1, keepdim=True)
        maxt, _ = torch.max(t,dim=1, keepdim=True)
        if any(torch.flatten(mint==maxt)):
            # print("a row have all the same elements, cannot use minmax normalization")
            ind = torch.flatten(mint!=maxt)
            t[ind] = (t[ind]-mint[ind])/(maxt[ind]-mint[ind])*(b-a)+a
            t[~ind] = torch.zeros_like(t[~ind])
            return t
        else:
            return (t-mint)/(maxt-mint)*(b-a)+a
    else:
        print("not in consideration, function name: minmaxnorm_torchtensor")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def duration2weight_graphBased(d, batch):
    pass
