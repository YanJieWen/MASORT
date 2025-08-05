'''
@File: track.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 16, 2025
@HomePage: https://github.com/YanJieWen
'''
'''
@File: kftrack.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 6月 29, 2025
@HomePage: https://github.com/YanJieWen
'''
import numpy as np
from copy import deepcopy
# from .mkalman_filter import MesKalmanFilterNew
from .kalman_filter import KalmanFilter
from .matching_iou import iou_batch

class TrackState:
    '''
    1:The trajectory just created is considered as a candidate state
    2:After hitting n frames in a row, it turns to the determined state
    3:If m consecutive frames are not shoted, the state will be changed to deletion state.
    '''
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class KalmanTrack(object):
    def __init__(self,bbox,score,track_id,n_init,max_age,delta_t=3,feat=None):
        #Initial kf for each track. That's to say Each trajectory has
        # a unique kf because it preserves its own measurements
        self.kf = KalmanFilter(dimx=8,dimz=4)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self.kf.F = np.array(
            [
                [1,0,0,0,1,0,0,0],
                [0,1,0,0,0,1,0,0],
                [0,0,1,0,0,0,1,0],
                [0,0,0,1,0,0,0,1],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
            ]
        )
        self.kf.H = np.array(
            [
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
            ]
        )
        xyah = self.xyxy_to_xyah(bbox)#(4,1)
        std = [
            2 * self._std_weight_position * xyah[3,0],
            2 * self._std_weight_position * xyah[3,0],
            1e-2,
            2 * self._std_weight_position * xyah[3,0],
            10 * self._std_weight_velocity * xyah[3,0],
            10 * self._std_weight_velocity * xyah[3,0],
            1e-5,
            10 * self._std_weight_velocity * xyah[3,0]]
        self.kf.P = np.diag(np.square(std))

        self.kf.x[:4] = xyah
        self.track_id = track_id
        self.score = score

        #Association params
        self.state = TrackState.Tentative
        self.time_since_update = 0
        self.hits = 1 # The cumulative number of frames of this trajectory
        self.age = 1
        self.hits_streak = 1 # Cumulative number of frames of fragmented trajectories
        self.delta_t = delta_t
        self._max_age = max_age
        self._n_init = n_init
        self.feats = []
        if feat is not None:
            self.feats.append(feat)

        #save historical observation
        self.history = [] #save prediction of kf
        self.last_observation = np.array([-1,-1,-1,-1]) # Placeholder
        self.history_observation = [] #save measurement only is not None
        self.observations = dict() #save measurement as dict only is not None
        self.velocity = None
        self.speed = None
        #todo: what's mean of this params?
        self.frozen = False #If True,
        # the bounding box has no observations and the speed of w and h is reset to 0.

    def predict(self):
        #negative bounding box is not allowed
        if self.kf.x[2]+self.kf.x[6]<=0:
            self.kf.x[6]=0
        if self.kf.x[3]+self.kf.x[7]<=0:
            self.kf.x[7]=0
        if self.frozen:
            self.kf.x[6] = self.kf.x[7] = 0
        Q = self.kf_process_noise(self.kf.x[3,0])

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update>0:
            self.hits_streak = 1 #if lost re-set 0
        self.time_since_update += 1
        self.history.append(self.tlbr)
        return self.history[-1] #(1,4)

    def update(self,bbox,embs=None,score=None):#(x,y,x,y)
        if bbox is not None:
            self.frozen = False
            if self.last_observation.sum()>=0:
                previous_box = None
                for dt in range(self.delta_t,0,-1):
                    if self.age-dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                #Calculate the motion direction of the trajectory based on
                # the historical bounding box and the current measurement
                self.velocity,self.speed = self.speed_prediction(previous_box,bbox)
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observation.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hits_streak += 1

            R = self.kf_mesurement_noise(self.kf.x[3,0])
            self.kf.update(self.xyxy_to_xyah(bbox),R=R)
            self.feats.append(embs)
            self.score = score
        else:
            self.kf.update(bbox)
            self.frozen = True

        if self.state == TrackState.Tentative and self.hits>self._n_init:
            self.state = TrackState.Confirmed
    #todo: CMC Calibration Observation
    def apply_affine_correction(self,affine):
        m = affine[:,:2] #2x2
        t = affine[:,2:] #2x1
        #for MPG
        if self.last_observation.sum()>0:
            ps = self.last_observation.reshape(2,2).T
            ps = m@ps+t
            self.last_observation = ps.T.reshape(-1)
        # for each obs box for MME
        for dt in range(self.delta_t,-1,-1):
            if self.age-dt in self.observations:
                ps = self.observations[self.age-dt].reshape(2,2).T
                ps = m@ps+t
                self.observations[self.age-dt] = ps.T.reshape(-1)
        #for kf state (x,p,last_obs)
        self.kf.apply_affine_correction(m,t)


    def mahalanobis(self,bbox):
        '''
        Run after predict() to calculate accuracy
        '''
        R = self.kf_mesurement_noise(self.kf.x[3,0])
        return self.kf.gating_distance(self.xyxy_to_xyah(bbox),R=R)

    def mark_removed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update>self._max_age:
            self.state = TrackState.Deleted

    @staticmethod
    def xyxy_to_xywh(xyxy):
        w = xyxy[2]-xyxy[0]
        h = xyxy[3]-xyxy[1]
        x = xyxy[0]+w/2.0
        y = xyxy[1]+h/2.0
        return np.array([x,y,w,h]).reshape((4,1))

    @staticmethod
    def xyxy_to_xyah(xyxy):
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        x = xyxy[0]+w/2.0
        y = xyxy[1]+h/2.0
        a = w/h
        return np.array([x,y,a,h]).reshape((4,1))

    @staticmethod
    def kf_process_noise(h,p=1./20,v=1./160):
        Q = np.diag(
            (
                (p * h) ** 2,
                (p * h) ** 2,
                1e-2 ** 2,
                (p * h) ** 2,
                (v * h) ** 2,
                (v * h) ** 2,
                1e-5 ** 2,
                (v * h) ** 2,
            )
        )
        return Q

    @staticmethod
    def kf_mesurement_noise(h,m=1./20):
        std = [
            m * h,
            m * h,
            1e-1,
            m * h]
        R = np.diag(np.square(std))
        return R

    @property
    def tlbr(self):
        ret = deepcopy(self.kf.x)
        x,y,a,h = ret.reshape(-1)[:4]
        w = a*h
        return np.array([x-w/2,y-h/2,x+w/2,y+h/2]).reshape(1,4)

    @property
    def tlwh(self):
        ret = deepcopy(self.kf.x)
        x,y,a,h = ret.reshape(-1)[:4]
        w = a*h
        return np.array([x-w/2,y-h/2,w,h])

    @staticmethod
    def speed_prediction(bbox1,bbox2):
        cx1,cy1 = (bbox1[0]+bbox1[2])/2,(bbox1[1]+bbox1[3])/2
        cx2,cy2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
        speed = np.array([cy2-cy1,cx2-cx1])
        norm = np.sqrt((cy2-cy1)**2+(cx2-cx1)**2)+1e-6

        level = 1-iou_batch(np.array([bbox1]),np.array([bbox2]))[0][0]
        return speed/norm,level

    @property
    def is_tentetive(self):
        return self.state==TrackState.Tentative

    @property
    def is_confirmed(self):
        return self.state ==TrackState.Confirmed

    @property
    def is_removed(self):
        return self.state==TrackState.Deleted


