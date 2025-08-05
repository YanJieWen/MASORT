'''
@File: matching_iou.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 02, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np

def iou_batch(bboxes1,bboxes2):
    bboxes1 = np.expand_dims(bboxes1,1)#MX1x4
    bboxes2 = np.expand_dims(bboxes2,0)#1XNx4
    xx1 = np.maximum(bboxes1[...,0],bboxes2[...,0])
    yy1 = np.maximum(bboxes1[...,1],bboxes2[...,1])
    xx2 = np.minimum(bboxes1[...,2],bboxes2[...,2])
    yy2 = np.minimum(bboxes1[...,3],bboxes2[...,3])
    w = np.maximum(0.,xx2-xx1)
    h = np.maximum(0.,yy2-yy1)
    wh = w*h
    area_bboxes1 = (bboxes1[...,2]-bboxes1[...,0])*(bboxes1[...,3]-bboxes1[...,1])
    area_bboxes2 = (bboxes2[...,2]-bboxes2[...,0])*(bboxes2[...,3]-bboxes2[...,1])
    union_area = area_bboxes1+area_bboxes2-wh
    iou_cost = wh/union_area
    return iou_cost




def iou_cost(tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    trks = np.asarray([tracks[tid].tlbr[0] for tid in track_indices])
    dets = np.asarray([detections[did].tlbr for did in detection_indices])
    iou_matrix = iou_batch(trks,dets)
    return iou_matrix


def _iou_cost(tracks,detections,track_indices=None,detection_indices=None):
    return 1-iou_cost(tracks,detections,track_indices,detection_indices)


def fuse_cost(tracks,detections,track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    iou_sim = iou_cost(tracks,detections,track_indices,detection_indices)
    det_scores = np.array([detections[idx].score for idx in detection_indices])
    det_scores = np.expand_dims(det_scores,axis=0).repeat(iou_sim.shape[0],axis=0)
    fuse_sim = iou_sim*det_scores
    fuse_cost = 1-fuse_sim
    return fuse_cost