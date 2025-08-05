'''
@File: diff_tools.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 02, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np
from copy import deepcopy



def pre_match_process(motion_results,appearance_results):
    '''
    将运动和外观的匹配进行分组，组合为匹配集合并输出未匹配轨迹和检测
    '''
    if len(motion_results[0]):
        motion_ma = np.asarray(motion_results[0])
        if len(appearance_results[0]):
            appear_ma = np.asarray(appearance_results[0])
            #sign: motion->0,appear->1
            pre_matches = np.vstack((np.insert(motion_ma,2,0,axis=1),np.insert(appear_ma,2,1,axis=1)))
        else:
            pre_matches = np.insert(motion_ma,2,0,axis=1)
        unmatched_tracks = list(set(motion_results[1]).intersection(set(appearance_results[1])))
        unmatched_detections = list(set(motion_results[2]).intersection(set(appearance_results[2])))
    else:
        pre_matches = np.asarray(appearance_results[0])
        if len(pre_matches)==0:
            pass
        else:
            pre_matches = np.insert(pre_matches,2,1,axis=1)
        unmatched_tracks = appearance_results[1]
        unmatched_detections = appearance_results[2]
    return pre_matches,unmatched_tracks,unmatched_detections

def get_conflict_sets(pre_matches):
    '''
    获得匹配的集合以及冲突的集合，即存在任何一个元素相等二另外一个元素不相等则视为冲突匹配
    '''
    t_vals, t_indexes, t_counts = get_unique_index(pre_matches[:, 0])
    d_vals, d_indexes, d_counts = get_unique_index(pre_matches[:, 1])
    matches,matches_ind = [],[]
    single_idxs = np.where(t_counts==1)[0]
    for i in single_idxs:
        t_ind = int(t_indexes[i])
        dv = pre_matches[t_ind,1]
        c = d_counts[np.argwhere(d_vals==dv)]
        if c==1:
            matches.append(pre_matches[t_ind,:])
            matches_ind.append(t_ind)
    two_idxs = np.where(t_counts==2)[0]
    for i in two_idxs:
        t_ind1 = int(t_indexes[i][0])
        t_ind2 = int(t_indexes[i][1])
        dv0 = pre_matches[t_ind1,1]
        dv1 = pre_matches[t_ind2,1]
        if dv0==dv1:
            matches.append(pre_matches[t_ind1, :])
            matches_ind.append(t_ind1)
            matches_ind.append(t_ind2)
    conflict_ind = list(set(np.arange(pre_matches.shape[0]))-set(matches_ind))
    conflict_matches = pre_matches[conflict_ind, :]
    return [tuple(x[:2]) for x in matches],conflict_matches

def get_unique_index(arr):
    '''
    按序列的track/det集合索引，原序列的索引集合，对应的元素数目
    '''
    sort_inds = np.argsort(arr)
    arr = np.asarray(arr)[sort_inds]
    vals, first_inds, _, counts = np.unique(arr, return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_inds, first_inds[1:])
    for x in indexes:
        x.sort()
    return vals,indexes,counts


def min_cost_conflict_matching(tracks,conflicts,alpha_gate):
    if conflicts.shape[0]==0:
        return [],[],[]
    speeds = np.asarray([tracks[tid].speed if tracks[tid].speed is not None else 0 for tid in conflicts[:,0]])
    conflicts = np.hstack((conflicts,speeds.reshape(-1,1)))
    first_round = conflicts[conflicts[:, -1] >= alpha_gate, :]
    matches_a = first_round[first_round[:,2]==1,:]

    second_round = conflicts[conflicts[:,-1]<alpha_gate,:]
    second_round = second_round[second_round[:,2]==0,:]
    #There cannot be duplicate matches in matches_a in second_round dets and tracks
    second_round = select_paris(second_round,0,matches_a[:,0])
    second_round = select_paris(second_round,1,matches_a[:,1])
    matches_b = second_round[:,:]
    matches_ = np.vstack((matches_a, matches_b))
    matches_ = matches_[:, :2].astype(int)
    unmatched_tracks = list(set(conflicts[:, 0]) - set(matches_[:, 0]))
    unmatched_tracks = [int(x) for x in unmatched_tracks]
    unmatched_detections = list(set(conflicts[:, 1]) - set(matches_[:, 1]))
    unmatched_detections = [int(x) for x in unmatched_detections]
    return list(map(tuple,matches_)),unmatched_tracks,unmatched_detections


def select_paris(matches,col_id,paris):
    '''
    基于运动的匹配中跟踪或检测索引不能包含于外观匹配中
    '''
    for t in paris:
        ind = np.argwhere(matches[:,col_id]==t)
        if len(ind):
            matches[ind[0],2] =-1
    matches = matches[matches[:,2]==0,:]
    return matches


def set_filter_matching(tracks,motion_results,appearance_results,alpha_gate):
    unmatched_tracks = list(set(motion_results[1]).intersection(set(appearance_results[1])))
    unmatched_detections = list(set(motion_results[2]).intersection(set(appearance_results[2])))
    matches_a = list(set(motion_results[0]).intersection(set(appearance_results[0])))
    motion_lefts = list(set(motion_results[0]) - set(matches_a))
    appear_lefts = list(set(appearance_results[0]) - set(matches_a))
    single_matches_motion = []
    for m in motion_lefts:
        appea_sp = list(zip(*appear_lefts))
        if len(appea_sp):
            if m[0] not in appea_sp[0] and m[1] not in appea_sp[1]:
                single_matches_motion.append(m)
        else:
            single_matches_motion.append(m)
    single_matches_appear = []
    for m in appear_lefts:
        motion_sp = list(zip(*motion_lefts))
        if len(motion_sp):
            if m[0] not in motion_sp[0] and m[1] not in motion_sp[1]:
                single_matches_appear.append(m)
        else:
            single_matches_appear.append(m)
    matches_a += (single_matches_appear + single_matches_motion)
    motion_lefts = np.asarray(list(set(motion_lefts) - set(single_matches_motion)))
    appear_lefts = np.asarray(list(set(appear_lefts) - set(single_matches_appear)))

    if motion_lefts.shape[0]:
        if appear_lefts.shape[0]:
            motion_lefts = np.insert(motion_lefts, 2, 0, axis=1)
            appear_lefts = np.insert(appear_lefts, 2, 1, axis=1)
            motion_lefts = motion_lefts[np.argsort(motion_lefts[:, 0])]
            appear_lefts = appear_lefts[np.argsort(appear_lefts[:, 0])]
            conflicts = np.vstack((motion_lefts, appear_lefts))
        else:
            conflicts = np.insert(motion_lefts,2,0,axis=1)
    else:
        conflicts = appear_lefts
        if len(conflicts)==0:
            return matches_a,unmatched_tracks,unmatched_detections
        else:
            conflicts = np.insert(conflicts,2,1,axis=1)

    speeds = np.asarray([tracks[tind].speed if tracks[tind].speed is not None else 0 for tind in conflicts[:,0]])
    conflicts = np.hstack((conflicts, speeds.reshape(-1, 1)))
    first_round = conflicts[conflicts[:, 3] >= alpha_gate, :]
    matches_b = first_round[first_round[:, 2] == 1,:]
    #todo:update track state
    for tid in matches_b[:, 0]:
        tracks[int(tid)].conflict_sign = 1
    second_round = conflicts[conflicts[:, 3] < alpha_gate, :]
    second_round = second_round[second_round[:, 2] == 0,:]
    _second_round = []
    for sec in second_round:
        if sec[0] not in matches_b[:, 0] and sec[1] not in matches_b[:, 1]:
            _second_round.append((int(sec[0]), int(sec[1])))
    for (tid,did) in _second_round:
        tracks[tid].conflict_sign = 2
    matches_b = list(map(tuple, matches_b[:, :2].astype(int)))
    matches_b += _second_round
    #alleviate empty set, if empty-->(0,2)
    matches_b = np.asarray(matches_b,dtype=int).reshape(-1,2)
    unmatched_tracks_conflicts = list(set(conflicts[:, 0]) - set(matches_b[:,0]))
    unmatched_detections_conflicts = list(set(conflicts[:, 1]) - set(matches_b[:,1]))
    unmatched_tracks_conflicts = [int(x) for x in unmatched_tracks_conflicts]
    unmatched_detections_conflicts = [int(x) for x in unmatched_detections_conflicts]
    unmatched_tracks += unmatched_tracks_conflicts
    unmatched_detections += unmatched_detections_conflicts

    matches_b = list(map(tuple,matches_b))
    matches = matches_a + matches_b
    return matches,unmatched_tracks,unmatched_detections