'''
@File: detection.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 6月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
import numpy as np


class Detection(object):
    def __init__(self,tlbr,score,feat):
        self.tlbr = np.asarray(tlbr,dtype=float)
        self.score = score
        self.feature = np.asarray(feat,dtype=float)

    @property
    def xywh(self):
        ret = self.tlbr.copy()
        ret[:2] += ret[2:]/2
        ret[2:] = self.tlbr[2:]-self.tlbr[:2]
        return ret

    @property
    def tlwh(self):
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret



