'''
@File: convert_dance_to_coco.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 6月 10, 2025
@HomePage: https://github.com/YanJieWen
'''


import os
import numpy as np
import json
import cv2

from tqdm import tqdm
import sys
import emoji

DATA_PATH = './datasets/dancetrack'
OUT_PATH = os.path.join(DATA_PATH,'annotations')
SPLITS = ['train','val','test']
#<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
if __name__ == '__main__':
    os.makedirs(OUT_PATH,exist_ok=True)
    for split in SPLITS:
        data_path = os.path.join(DATA_PATH,split)
        out_path = os.path.join(OUT_PATH,f'{split}.json')
        out = {'images':[],'annotations':[],'videos':[],
               'categories':[{'id':1,'name':'dancer'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        pbar = tqdm(sorted(seqs),desc=f'{emoji.emojize(":down_arrow:")}: Cache annotations....',file=sys.stdout)
        for seq in pbar:
            if '.DS_Store' in seq or '.ipy' in seq:
                continue
            video_cnt += 1
            pbar.desc = f'{emoji.emojize(":rocket:")}: Reading seq of {seq}'
            out['videos'].append({'id':video_cnt,'file_name':seq})
            seq_path = os.path.join(data_path,seq)
            img_path = os.path.join(seq_path,'img1')
            ann_path = os.path.join(seq_path,'gt/gt.txt')
            images = os.listdir(img_path)
            num_images = len([img for img in images if img.endswith('.jpg')])

            for i in range(num_images):
                imp = os.path.join(img_path,f'{str(i+1).zfill(8)}.jpg')
                assert os.path.isfile(imp),f'{emoji.emojize(":prohibited:")}:{imp} is not exists'
                img = cv2.imread(imp)
                height,width = img.shape[:2]
                image_info = {
                    'file_name': f'{seq}/img1/{str(i+1).zfill(8)}.jpg',
                    'id': image_cnt+i+1,
                    'frame_id': i+1,
                    'prev_image_id': image_cnt+i if i >0 else -1,
                    'nex_image_id': image_cnt+i+2 if i<num_images-1 else -1,
                    'video_id': video_cnt,
                    'height': height,
                    'width': width,
                }
                out['images'].append(image_info)
            if split!='test':
                anns = np.loadtxt(ann_path,dtype=np.float32,delimiter=',')
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    category_id = 1
                    ann = {
                        'id':ann_cnt,
                        'category_id':category_id,
                        'image_id':image_cnt+frame_id,
                        'track_id':track_id,
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]),
                        'iscrowd':0,
                        'area':float(anns[i][4]*anns[i][5])
                    }
                    out['annotations'].append(ann)
                print(f'{emoji.emojize(":flag_china:")}:{seq}-{int(anns[:,0].max())} ann images')
            image_cnt += num_images
        print(f'{emoji.emojize(":dog_face:")*3} loaded {split} for '
              f'{len(out["images"])} images and {len(out["annotations"])} instances')
        json.dump(out,open(out_path,'w'))


