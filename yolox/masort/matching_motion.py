'''
@File: matching_motion.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 01, 2025
@HomePage: https://github.com/YanJieWen
'''


import numpy as np

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1]
    for dt in range(k,0,-1):
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]

def speed_direction_batch(dets,tracks):
    tracks = tracks[...,np.newaxis]#(MX4X1)
    CX1,CY1 = (dets[:,0]+dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2,CY2 = (tracks[:,0]+tracks[:,2])/2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1-CX2#MXN
    dy = CY1-CY2
    norm = np.sqrt(dx**2+dy**2)+1e-6
    dx = dx/norm
    dy = dy/norm
    return dx,dy


def motion_cost(tracks,detections,track_indices=None,detection_indices=None,vdc_weight=0.2,delta_t = 3):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    #tracklets speed
    velocities = np.array([tracks[tid].velocity if tracks[tid].velocity is not None else np.array((0,0)) for tid in track_indices])
    previous_obs = np.array([k_previous_obs(tracks[tid].observations, tracks[tid].age, delta_t) for tid in track_indices])
    trks = np.asarray([tracks[tid].tlbr[0] for tid in track_indices])#MX4
    dets = np.asarray([detections[did].tlbr for did in detection_indices])
    #Calculate the directional consistency of the current detection and trajectory
    X,Y = speed_direction_batch(dets,previous_obs)#MXN,MXN
    inertial_Y, inertial_X = velocities[:,0],velocities[:,1]#M,M
    inertial_Y = np.repeat(inertial_Y[:,np.newaxis],Y.shape[1],axis=1)
    inertial_X = np.repeat(inertial_X[:,np.newaxis],X.shape[1],axis=1)
    diff_angle_cos = np.clip(inertial_X*X+inertial_Y*Y,a_min=-1,a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi/2.0-np.abs(diff_angle))/np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,3]<0)] = 0 #delat_t may be initial M previous_obs or trks?-->NO DIFFERENT
    scores = np.asarray([detections[idx].score for idx in detection_indices])
    scores = np.repeat(scores[np.newaxis],trks.shape[0],axis=0)
    valid_mask = np.repeat(valid_mask[:,np.newaxis],X.shape[1],axis=1)#MXN
    angle_diff_cost = ((valid_mask*diff_angle)*vdc_weight)*scores
    return angle_diff_cost
#todo:ablation-2
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}
INFTY_COST = 1e+5
def maha_cost(tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    gating_thresh = chi2inv95[4]
    cost_matrix = np.zeros((len(track_indices),len(detection_indices)))
    for row,track_idx in enumerate(track_indices):
        for col,det_idx in enumerate(detection_indices):
            track = tracks[track_idx]
            det = detections[det_idx]
            distance = track.mahalanobis(det.tlbr)
            cost_matrix[row,col] = distance
    cost_matrix[cost_matrix>gating_thresh] = INFTY_COST
    return cost_matrix

