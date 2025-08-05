'''
@File: visual_sample.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 6月 10, 2025
@HomePage: https://github.com/YanJieWen
'''

import cv2
from pycocotools.coco import COCO
from PIL import Image
import PIL.ImageDraw as Imagedraw
from PIL import ImageColor,ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial


np.random.seed(42)
COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen']

COLORS = np.random.permutation(COLORS)
DATA_PATH = 'datasets/dancetrack'
SPLITS = ['train','val','test']
DATA_ROOT = os.path.join(DATA_PATH,'annotations')
OUT_ROOT = os.path.join(DATA_PATH,'demo')


def visual(id,img_path,split,vid):
    anns = coco.loadAnns(coco.getAnnIds(id))
    ann_infos = [(x['bbox'], x['track_id']) for x in anns]
    if split != 'test':
        bboxes, tids = zip(*ann_infos)
        if 'mot' in DATA_PATH or 'MOT' in DATA_PATH:
            _split = 'train'
        else:
            _split = split
        img = Image.open(os.path.join(DATA_PATH,_split,img_path)).convert('RGB')
        bboxes = np.asarray(bboxes)
        bboxes[:, 2:] += bboxes[:, :2]
        tids = np.asarray(tids, dtype=int)
        colorss = [ImageColor.getrgb(COLORS[tr % len(COLORS)]) for tr in tids]
        draw = Imagedraw.Draw(img)
        for box, tid, color in zip(bboxes, tids, colorss):
            left, top, right, bottom = box
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=5, fill=color)
            font = ImageFont.truetype('arialbd.ttf', 32)
            display_str = f"ID:{tid}"
            display_str_heights = [font.getsize(ds)[1] for ds in display_str]
            display_str_height = (1 + 2 * 0.05) * max(display_str_heights)
            if top > display_str_height:
                text_top = top - display_str_height
                text_bottom = top
            else:
                text_top = bottom
                text_bottom = bottom + display_str_height
            for ds in display_str:
                text_width, text_height = font.getsize(ds)
                margin = np.ceil(0.05 * text_width)
                draw.rectangle([(left, text_top),(left + text_width + 2 * margin, text_bottom)], fill=color)
                draw.text((left + margin, text_top),
                          ds,
                          fill='black',
                          font=font)
                left += text_width
    else:
        img = Image.open(os.path.join(DATA_PATH, split, img_path)).convert('RGB')
    img = np.array(img)[..., ::-1]
    cv2.imwrite(os.path.join(OUT_ROOT,split,f'{vid}_{os.path.basename(img_path)}'), img)

if __name__ == '__main__':
    os.makedirs(OUT_ROOT,exist_ok=True)
    for split in SPLITS:
        demo_root = os.path.join(OUT_ROOT,split)
        os.makedirs(demo_root,exist_ok=True)
        coco = COCO(os.path.join(DATA_ROOT,f'{split}.json'))
        #将图像信息按照序列保存
        img_ids = sorted(coco.getImgIds())
        img_infos = coco.loadImgs(img_ids)
        seq_infos = {}
        for img_info in img_infos:
            seq_infos.setdefault(img_info['video_id'],[]).append(img_info)
        seq_keys = list(seq_infos.keys())
        selected_keys = seq_keys[:2]+seq_keys[-2:]
        for key in selected_keys:
            seq_info = seq_infos[key]
            info = [(img['id'],img['file_name']) for img in seq_info]
            s_ids,s_imgs = zip(*info[::10][:3])
            e_ids,e_imgs = zip(*info[::10][-3:])
            vis_func = partial(visual,split=split,vid=key)
            list(map(vis_func,list(s_ids),list(s_imgs)))
            list(map(vis_func,list(e_ids),list(e_imgs)))
            print(f'Demo {split}--seq {key}')

