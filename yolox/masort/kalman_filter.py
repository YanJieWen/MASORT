'''
@File: kalman_filter.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 16, 2025
@HomePage: https://github.com/YanJieWen
'''
from copy import deepcopy
from math import log,exp,sqrt
import sys
import numpy as np
import scipy.linalg

from filterpy.stats import logpdf
from filterpy.common import reshape_z,pretty_str


class KalmanFilter(object):
    def __init__(self,dimx,dimz,dimu=0):
        if dimx < 1:
            raise ValueError('dimx must be 1 or greater')
        if dimz < 1:
            raise ValueError('dimz must be 1 or greater')
        self.dim_x = dimx
        self.dim_z = dimz
        self.dim_u = dimu
        self.x = np.zeros((dimx, 1))  # kf state-->[x,y,w,h,x',y',w',h']
        self.P = np.eye(dimx)  # covariance
        self.Q = np.eye(dimx)  # noise matrix
        self.B = None  # control transition matrix
        self.F = np.eye(dimx)  # state transition matrix
        self.R = np.eye(dimz)  # observed noise
        self.H = np.zeros((dimz, dimx))  # observed function
        self.z = np.array([[None] * self.dim_z]).T  # measurement-->[x,y,w,h]
        self.K = np.zeros((dimx, dimz))  # kalman gain
        self.y = np.zeros((dimz, 1))  # residual
        self.S = np.zeros((dimz, dimz))  # system covariance
        self._I = np.eye(dimx)

    def predict(self,u=None,B=None,F=None,Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q
        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)
        self.P = np.linalg.multi_dot((F, self.P, F.T)) + Q

    def project(self,mean,covariance,R,H):
        mean = np.dot(H,mean)
        covariance = np.linalg.multi_dot((H,covariance,H.T))
        return mean,covariance+R

    def update(self,z,R=None,H=None):
        if z is None:
            return
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H
        pro_x,pro_p = self.project(self.x,self.P,R,H)
        self.S = pro_p.copy()
        self.y = z-pro_x

        chol_factor,lower = scipy.linalg.cho_factor(self.S,lower=True,check_finite=False)
        #(dimx,dimz)
        self.K = scipy.linalg.cho_solve((chol_factor,lower),np.dot(self.P,H.T).T,check_finite=False).T
        self.x += np.dot(self.K,self.y)
        I_KH = self._I-np.dot(self.K,H)
        self.P = np.linalg.multi_dot((I_KH,self.P,I_KH.T))+np.linalg.multi_dot((self.K,R,self.K.T))
        self.z = deepcopy(z)

    def gating_distance(self,z,R=None):
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R
        pro_x,pro_p = self.project(self.x,self.P,R=R,H=self.H)
        d = z-pro_x #dimzx1
        cholesky_factor = np.linalg.cholesky(pro_p)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d, lower=True, check_finite=False,
            overwrite_b=True
        )
        squared_maha = float(np.sum(z * z, axis=0))
        return squared_maha


