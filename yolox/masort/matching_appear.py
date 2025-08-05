'''
@File: matching_appear.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 01, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np
import torch

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x,y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

def _res_recons_cosing_distance(x,y,tmp=100):
    x = np.asarray(x) /np.linalg.norm(x,axis=1,keepdims=True)
    y = np.asarray(y) / np.linalg.norm(y,axis=1,keepdims=True)
    ftrk = torch.from_numpy(x).half().cuda()#MXD
    fdet = torch.from_numpy(y).half().cuda()#NXD
    aff = torch.mm(ftrk,fdet.transpose(0,1)) #MXN
    aff_fd = torch.nn.functional.softmax(tmp*aff,dim=1)#MXN
    aff_dt = torch.nn.functional.softmax(tmp*aff,dim=0).transpose(1,0)#NXM
    res_recons_ftrk = torch.mm(aff_fd,fdet)#MXD
    res_recons_fdet = torch.mm(aff_dt,ftrk)#NXD
    sim = (aff+torch.mm(res_recons_ftrk,res_recons_fdet.transpose(0,1)))/2
    distance = 1-sim
    distance = distance.detach().cpu().numpy()
    return distance

APPER_DIST = {
    'cosin': _cosine_distance,
    'attn': _res_recons_cosing_distance,
}

class NearstNeighborDistanceMetric(object):
    def __init__(self,dist_type='cosin',alpha=0.9,det_thresh=0.7,w_assoc_emb=0.75,aw_param=0.5):
        self._metric = APPER_DIST[dist_type]
        self.samples = {}
        self.alpha = alpha
        self.det_thresh = det_thresh
        self.w_assoc_emb = w_assoc_emb
        self.aw_alpha = aw_param
    # We use exponential averaging to update appearance features
    # to avoid long-term feature contamination and discard the budget function
    def partial_fit(self,features,targets,scores,active_targets):
        for feat,tgt,conf in zip(features,targets,scores):
            #todo: EA->frozen EMA update

            trust = (conf-self.det_thresh)/(1-self.det_thresh)
            det_alpha = self.alpha+(1-self.alpha)*(1-trust)
            # det_alpha = self.alpha
            if tgt not in self.samples:
                self.samples[tgt] = feat
            else:
                self.samples[tgt] = det_alpha*self.samples[tgt]+(1-det_alpha)*feat
                #todo:if need
                self.samples[tgt] /= np.linalg.norm(self.samples[tgt])
        #to ensure lost is retain and filter removed
        self.samples = {k:self.samples[k] for k in active_targets}#only confirmed is saved
    def distance(self,tracks,detections,track_indices=None,detection_indices=None):
        if track_indices is None:
            track_indices = list(range(len(tracks)))
        if detection_indices is None:
            detection_indices = list(range(len(detections)))
        features = np.array([detections[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        all_trfeats = np.stack([self.samples[tgt] for tgt in targets],axis=0)#MXD
        cost_matrix = self._metric(all_trfeats,features)
        return cost_matrix
    def boost_distance(self,tracks,detections,track_indices=None,detection_indices=None):
        if track_indices is None:
            track_indices = list(range(len(tracks)))
        if detection_indices is None:
            detection_indices = list(range(len(detections)))
        det_embs = np.array([detections[i].feature for i in detection_indices])
        tgts = np.array([tracks[i].track_id for i in track_indices])
        trk_embs = np.stack([self.samples[tgt] for tgt in tgts],axis=0)
        emb_cost = None if (det_embs.shape[0]==0 or trk_embs.shape[0]==0) else 1.-self._metric(trk_embs,det_embs)
        if emb_cost is None:
            emb_cost = 0
        else:
            pass
        #TODO: AE-->frozen adptive boosting
        w_matrix = self.boost_emb_alpha(emb_cost,self.w_assoc_emb,self.aw_alpha)
        return emb_cost*w_matrix

        # return emb_cost*self.w_assoc_emb
    @staticmethod
    def boost_emb_alpha(emb_cost,w_association_emb,max_diff=0.5):
        w_emb = np.full_like(emb_cost,w_association_emb)
        w_emb_bouns = np.full_like(emb_cost,0)

        if emb_cost.shape[1]>=2:
            for idx in range(emb_cost.shape[0]):
                inds = np.argsort(-emb_cost[idx])
                row_weights = min(emb_cost[idx,inds[0]]-emb_cost[idx,inds[1]],max_diff)
                w_emb_bouns[idx] += row_weights/2
        if emb_cost.shape[0]>=2:
            for idx in range(emb_cost.shape[1]):
                inds = np.argsort(-emb_cost[:,idx])
                col_weights = min(emb_cost[inds[0],idx]-emb_cost[inds[1],idx],max_diff)
                w_emb_bouns[:,idx] += col_weights/2
        return w_emb+w_emb_bouns
