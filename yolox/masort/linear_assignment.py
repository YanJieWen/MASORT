'''
@File: linear_assignment.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 01, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from .matching_iou import iou_cost,_iou_cost
from .matching_motion import motion_cost,maha_cost


from .diff_tools import pre_match_process,get_conflict_sets,min_cost_conflict_matching,set_filter_matching

INFTY_COST = 1e+5

def _linear_assignment(cost_matrix):
    try:
        import lap
        _,x,y = lap.lapjv(cost_matrix,extend_cost=True)
        return [[y[i],i] for i in x if i >= 0]
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return list(zip(x,y))


def min_cost_matching(distance_metric,max_distance,tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    if len(detection_indices)==0 or len(track_indices)==0:
        return [],track_indices,detection_indices
    cost_matrix = distance_metric(tracks=tracks,detections=detections,
                                  track_indices=track_indices,detection_indices=detection_indices)
    cost_matrix[cost_matrix>max_distance] = max_distance+1e-5
    row_indices,col_indices = linear_assignment(cost_matrix)
    matches,unmatched_tracks,unmatched_detections = [],[],[]
    for col,detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row,track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row,col in zip(row_indices,col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row,col]>max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx,detection_idx))
    return matches,unmatched_tracks,unmatched_detections
#triple filter
def gate_cost_matching(distance_metric,args,tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(tracks))
    if detection_indices is None:
        detection_indices = list(range(detections))
    if len(detection_indices)==0 or len(track_indices)==0:
        return [],track_indices,detection_indices
    iou_matrix = iou_cost(tracks,detections,track_indices,detection_indices)
    #todo: mahacost
    motion_matrix = motion_cost(tracks, detections, track_indices, detection_indices, args.inertia, args.delta_t)
    # motion_matrix = -maha_cost(tracks,detections,track_indices,detection_indices)
    emb_matrix = distance_metric(tracks,detections,track_indices,detection_indices)
    a = np.asarray(iou_matrix>args.oc_match_thresh,dtype=np.int32)
    if a.sum(1).max() ==1 and a.sum(0).max()==1:
        matches = list(zip(*np.where(a)))
    else:
        matches = _linear_assignment(-(iou_matrix+motion_matrix+emb_matrix))
    unmatched_tracks,unmatched_detections = [],[]
    _matches = np.array(matches)
    for col,detection_idx in enumerate(detection_indices):
        if col not in _matches[:,1]:
            unmatched_detections.append(detection_idx)
    for row,track_idx in enumerate(track_indices):
        if row not in _matches[:,0]:
            unmatched_tracks.append(track_idx)
    matches = []
    for m in _matches:
        rowind,colind = m
        track_idx = track_indices[rowind]
        detection_idx = detection_indices[colind]
        if iou_matrix[rowind,colind]<args.oc_match_thresh:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx,detection_idx))
    return matches,unmatched_tracks,unmatched_detections

def motion_in_appearance(distance_metric,args,tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(tracks))
    if detection_indices is None:
        detection_indices = list(range(detections))
    if len(detection_indices)==0 or len(track_indices)==0:
        return [],track_indices,detection_indices

    emb_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    gated_threshold = 9.4877
    for row,track_idx in enumerate(track_indices):
        for col,det_idx in enumerate(detection_indices):
            gate_distance = tracks[track_idx].mahalanobis(detections[det_idx].tlbr)
            if gate_distance>gated_threshold:
                emb_matrix[row,col] = INFTY_COST
    row_indices, col_indices = linear_assignment(emb_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if emb_matrix[row, col] > args.appear_thresh:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

def appear_in_motion(distance_metric,args,tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(tracks))
    if detection_indices is None:
        detection_indices = list(range(detections))
    if len(detection_indices)==0 or len(track_indices)==0:
        return [],track_indices,detection_indices
    gated_threshold = 9.4877
    emb_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    motion_cost = np.zeros_like(emb_matrix)
    for row, track_idx in enumerate(track_indices):
        for col, det_idx in enumerate(detection_indices):
            gate_distance = tracks[track_idx].mahalanobis(detections[det_idx].tlbr)
            motion_cost[row,col] = gate_distance
    cost_matrix = gate_assignment(motion_cost,emb_matrix,args.appear_thresh)
    row_indices, col_indices = linear_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > gated_threshold:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

def gate_assignment(main_cost,second_cost,threshold,gated_cost=INFTY_COST):
    main_cost[second_cost>threshold] = gated_cost
    return main_cost

def union_min_cost_matching(tracks,detections,track_indices,detection_indices,nn_matcher,args):
    '''
    Parallel association based on motion level
    Args:
        tracks:List[class]
        detections:List[class]
        track_indices:List
        detection_indices:List
        args:embed_off/motion_off/union_off/appear_thresh=0.3/iou_thresh=0.3
    Returns: List: matches,unmatched_tracks,unmatched_detections
    '''
    if track_indices is None:
        track_indices = list(range(tracks))
    if detection_indices is None:
        detection_indices = list(range(detections))
    if len(detection_indices)==0 or len(track_indices)==0:
        return [],track_indices,detection_indices
    if args.union_off:
        if not args.motion_off:
            matches,unmatched_tracks,unmatched_detections = gate_cost_matching(nn_matcher.boost_distance,args,tracks,detections,track_indices,detection_indices)
        elif not args.embed_off:
            matches,unmatched_tracks,unmatched_detections = min_cost_matching(nn_matcher.distance,args.appear_thresh,tracks,detections,track_indices,detection_indices)
        else:
            matches, unmatched_tracks, unmatched_detections = min_cost_matching(_iou_cost,args.match_thresh,tracks,detections,track_indices,detection_indices)
    else: #run union-based assosiate through alpha_gate
        motion_pre_assign = gate_cost_matching(nn_matcher.boost_distance,args,tracks,detections,track_indices,detection_indices)
        appearance_pre_assign = min_cost_matching(nn_matcher.distance,args.appear_thresh,tracks,detections,track_indices,detection_indices)
        # pre_matches,unmatched_tracks,unmatched_detections = pre_match_process(motion_pre_assign,appearance_pre_assign)
        #todo: whta's differents between the original code-->the same

        matches, unmatched_tracks, unmatched_detections = set_filter_matching(tracks,motion_pre_assign,appearance_pre_assign,args.alpha_gate)
        #
        # matches,conflicts = get_conflict_sets(pre_matches)
        # _matches,_unmatched_tracks,_unmatched_detections = min_cost_conflict_matching(tracks,conflicts,args.alpha_gate)
        # matches += _matches
        # unmatched_detections += _unmatched_detections
        # unmatched_tracks += _unmatched_tracks
        #todo: series matching
        # matches, unmatched_tracks, unmatched_detections = appear_in_motion(nn_matcher.distance,args,tracks,detections,track_indices,detection_indices)

        matches = sorted(matches,key=lambda m:m[0])
    return matches,unmatched_tracks,unmatched_detections






