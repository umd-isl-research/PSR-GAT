import json
from sklearn.preprocessing import LabelEncoder
# import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from itertools import groupby
from util import noise_user_batch2
#np.random.seed(42)


def mergeSuccessive(a, b):
    """
    :param a: a list of node ids
    :param b: a list of duration corresponding to that nodes, len(a) = len(b)
    :return val, dur: with successive nodes removed and duration merged, e.g a=[2,3,3,4,2], b=[1.1,2.2,3.3,2.5,3.1], return val=[2,3,4,2] and dur=[1.1,5.5,2.5,3.1]
    """
    assert len(a)==len(b)
    length=len(a)
    if length<2:
        return a,b
    valnum=[(k,sum(1 for i in g)) for k,g in groupby(a)]
    val,dur=[],[]
    ind=0
    for v, num in valnum:
        val.append(v)
        dur.append(sum(b[ind:ind+num]))
        ind+=num
    return val, dur

def scr_sta_parse(usage_info):
    """
    :param usage_info: [[{},{}],[{}, {}],[{}, {}]], dict could be empty
    :return: scr_freq_m, scr_dur_m, scr_freq_w, scr_dur_w, scr_freq_d, scr_dur_d, type: list
    """
    num_scr = len(item_decoder)
    scr_freq_m, scr_dur_m, scr_freq_w, scr_dur_w, scr_freq_d, scr_dur_d = [0]*num_scr, [0]*num_scr, [0]*num_scr, [0]*num_scr, [0]*num_scr ,[0]*num_scr
    sta_m, sta_w, sta_d = usage_info

    if sta_m[0]:
        for scr, freq in sta_m[0].items():
            scr_freq_m[item_encoder[scr]] = freq
    if sta_m[1]:
        for scr,dur in  sta_m[1].items():
            scr_dur_m[item_encoder[scr]] = dur
    if sta_w[0]:
        for scr,freq in sta_w[0].items():
            scr_freq_w[item_encoder[scr]] = freq
    if sta_w[1]:
        for scr,dur in sta_w[1].items():
            scr_dur_w[item_encoder[scr]] = dur
    if sta_d[0]:
        for scr,freq in sta_d[0].items():
            scr_freq_d[item_encoder[scr]] = freq
    if sta_d[1]:
        for scr,dur in sta_d[1].items():
            scr_dur_d[item_encoder[scr]] = dur
    return scr_freq_m, scr_dur_m, scr_freq_w, scr_dur_w, scr_freq_d, scr_dur_d


class SYNCScrSessionDataset_v4(InMemoryDataset):  
    """
	this is stable by now(Mar-28-2021), for the task: predicting the last screen using all prior screen of a session
    The session data include statistics of screen usage in last month/week/day
	{userid: {sessionid:[[scr1,scr2,...],[dur1,dur2,...], [t1,t2,...], usageInfo]}}
	where usageInfo is a list: [[{scr:freq_month},{scr:dur_month}],[{scr:freq_week}, {scr:dur_week}],[{scr:freq_day}, {scr:dur_day}]]
    """
    def __init__(self, root="./SYNC_multiscale", transform=None, pre_transform=None, tar_uid=None):
        self.tar_uid = tar_uid
        super(SYNCScrSessionDataset_v4, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ["graph_binary_"+self.tar_uid+".dataset"]
    def download(self):
        pass
    def process(self):
        #last_scr_ana = {}
        data_list = []
        for session_id, (items, viewDuration, timestamp, usage_info) in tar_sessions.items():
            if len(items)<=2:
                continue
            items, viewDuration = mergeSuccessive(items, viewDuration)  # remove successive duplications
            if len(items)<3:  # at least two nodes in a graph and one as ground truth label,
                continue
            y = torch.FloatTensor([item_encoder[items[-1]]])    # use the last item in a session sequence to be ground truth
            item_id = [item_encoder[i] for i in items[:-1]]
            sess_item_id = LabelEncoder().fit_transform(item_id)  # in order to get index of source to target for edges
            source_nodes = sess_item_id[:-1]
            target_nodes = sess_item_id[1:]
            edge_index = torch.tensor([source_nodes,target_nodes], dtype=torch.long)
            edge_attr = torch.FloatTensor(viewDuration)  # here I do not want to convert duration to weight in data processing, I want to reserve the original value and convert it later
            s_edge_attr = edge_attr[:-2]  # edge attributes for edges between source and target nodes
            last2_edge_attr = edge_attr[-2:][None,:]  # the last two edge attributes
            x = item_id
            x = torch.LongTensor(x).unsqueeze(1)

            scr_freq_m, scr_dur_m, scr_freq_w, scr_dur_w, scr_freq_d, scr_dur_d = scr_sta_parse(usage_info)
            scr_freq_m, scr_dur_m, scr_freq_w, scr_dur_w, scr_freq_d, scr_dur_d = torch.FloatTensor(scr_freq_m)[None,:], torch.FloatTensor(scr_dur_m)[None,:], torch.FloatTensor(scr_freq_w)[None,:], torch.FloatTensor(scr_dur_w)[None,:], torch.FloatTensor(scr_freq_d)[None,:], torch.FloatTensor(scr_dur_d)[None,:]
            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=s_edge_attr, last2_edge_attr=last2_edge_attr, scr_freq_m=scr_freq_m, scr_dur_m=scr_dur_m, scr_freq_w=scr_freq_w, scr_dur_w=scr_dur_w, scr_freq_d=scr_freq_d, scr_dur_d=scr_dur_d)
            data_list.append(data)
        if len(data_list)==0:
            print("encountering a user does not have valid sessions")
            return  # in not return, errors will occur
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":  # this is to build graphs for the 3 months' data
    use_scr_view_duration = True  # we will always use view duration
    start_screen = ("HOME/Root", "ROOTSTATE/HmiBlackScreen", "ROOTSTATE/SuspendMode", "ROOTSTATE/SyncWithAl",
                    "ROOTSTATE/WelcomeAnimation", "ROOTSTATE/FarewellAnimationView", "STA")  # 7 default screens
    session_file = "..\\SYNC_preprocessedData\\batch2 sessions\\batch2_BCdivided_sessions_scrUsageStat.json"  # the path of the session file in your computer

    with open("..\\SYNC_preprocessedData\\batch2_all_screens.json", 'r') as f:
        all_screens = json.load(f)  # read all screen names
        all_screens = list(set(all_screens) - set(start_screen))  # remove defaults screen names
        print("all screens loaded!")  # this is private data

    with open(session_file, 'r') as f:  # load session data extracted from the raw dataset, sessions are extracted by Tian Qin
        all_sessions = json.load(f)
        print("all sessions loaded!")

    with open("..\\SYNC_preprocessedData\\batch2_all_users.json", 'r') as f:  # load all user hashed strings
        users = json.load(f)
        users = set(users) - set(noise_user_batch2)  # these users have noise data so removed
        print("all users loaded!")
    num_screens = len(all_screens)
    item_encoder = LabelEncoder()
    all_item_ids = item_encoder.fit_transform(sorted(all_screens))
    item_encoder = dict(zip(sorted(all_screens), all_item_ids))
    item_decoder = {v: k for k, v in item_encoder.items()}
    for tar_uid in users:
        tar_sessions = all_sessions[tar_uid]
        _ = SYNCScrSessionDataset_v4(root=".\\SYNC_batch2_BCdivided_multiscale_lastanchor", tar_uid=tar_uid)
