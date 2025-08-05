'''
@File: embedding.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 6月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
import pickle

import torch
import cv2
import torchreid
import numpy as np

import os
from collections import OrderedDict
import sys
sys.path.append('external/fast-reid/')
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg
class FastReID(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        yaml_path = './external/fast-reid/configs/mot17-sbs.yml'
        # config_file = "yolox/fashiontracker/fast_reid/configs/"
        self.cfg = setup_cfg(yaml_path, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def forward(self, batch):
        # Uses half during training
        batch = batch.half()
        with torch.no_grad():
            return self.model(batch)


class EmbddingComputer:
    def __init__(self,dataset,test_dataset,max_batch=1024):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128,384)
        os.makedirs('./cache/embeddings/',exist_ok=True)
        self.cache_path = './cache/embeddings/{}_embedding.pkl'
        self.cache = {}
        self.cache_name = ""
        self.max_batch = max_batch
        self.normalize = False #Fastreid is False, torchreid is True

    def initialize_model(self):
        if self.dataset=='mot17':
            if self.test_dataset:
                path = "pretrained/reid-model/mot17_sbs_S50.pth"
            else:
                return self._get_general_model() #torchreid
        elif self.dataset == 'mot20':
            if self.test_dataset:
                path = "pretrained/reid-model/mot20_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == 'dance':
            path = "pretrained/reid-model/dance_sbs_S50.pth"

        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model


    def load_cache(self,path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.isfile(cache_path):
            with open(cache_path,'rb') as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self,img,box,tag):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0]!=box.shape[0]:
                raise RuntimeError("ERROR: The number of cached embedding do not match the detections. Please Delete embedding cache")
            return embs

        if self.model is None:
            self.initialize_model()

        h,w = img.shape[:2]
        results = np.round(box).astype(np.int32)
        # Some boxes have the same upper left corner and lower right corner
        results[:, 0] = results[:, 0].clip(0, w - 1)
        results[:, 1] = results[:, 1].clip(0, h - 1)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)

        crops = []
        for p in results:
            #todo: may all x1=x2 or y1=y2
            if p[2]==p[0]:
                p[2] += 1
            if p[-1]==p[1]:
                p[-1] += 1
            crop = img[p[1]:p[3],p[0]:p[2]]
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if self.normalize:
                crop /= 255
                crop -= np.array((0.485, 0.456, 0.406))
                crop /= np.array((0.229, 0.224, 0.225))
            crop = torch.as_tensor(crop.transpose(2, 0, 1))
            crop = crop.unsqueeze(0)
            crops.append(crop)
        crops = torch.cat(crops,dim=0)

        embs = []
        for idx in range(0,len(crops),self.max_batch):
            batch_crops = crops[idx:idx+self.max_batch]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs,dim=-1)
        embs = embs.cpu().numpy()

        self.cache[tag] = embs
        return embs


    def _get_general_model(self):
        """Used for the half-val for MOT17/20.

        The MOT17/20 SBS models are trained over the half-val we
        evaluate on as well. Instead we use a different model for
        validation.
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("pretrained/reid-model/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        self.normalize = True

