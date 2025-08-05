'''
@File: mytrack.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 5月 18, 2025
@HomePage: https://github.com/YanJieWen
'''
import sys
sys.path.append("external")

from loguru import logger

import torch

from yolox.exp import get_exp
from yolox.utils import get_model_info,setup_logger
from yolox.evaluators import MOTEvaluator


import os
import os.path as osp
import shutil
import argparse
import random
from glob import glob
import motmetrics as mm
from pathlib import Path
from collections import OrderedDict
import emoji

from TrackEval import trackeval
import pandas as pd


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

''' 
From
https://github.com/FoundationVision/ByteTrack/blob/main/tools/track.py
This is a simplified version for bytetrack-Implementation
Ignore invalidation parameters
'''


parser = argparse.ArgumentParser("My Fashion YOLOX Eval with ByteTrack")
parser.add_argument('-b',"--batch-size",type=int,default=1,help='batch_size')
parser.add_argument('-d',"--devices",type=int,default=1,help='device for training')
parser.add_argument('-f',"--exp-file",type=str,default='./exps/example/mot/yolox_x_ablation.py')
parser.add_argument('-p',"--fp16",type=bool,default=False,help='Adopting mix precision eval')
parser.add_argument('-t','--test',default=False,help='Evaluating on test-dev set')
parser.add_argument('-o','--out-root',default='./masort/ablation',help='Logging output root')
#Detection args
parser.add_argument('-c',"--ckpt",default='./pretrained/masortweight/masort_ablation.pth.tar',help='Detector weight')
parser.add_argument("--conf", default=0.01, type=float, help="test conf")
parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
parser.add_argument("--tsize", default=None, type=int, help="test img size")
parser.add_argument("--seed", default=None, type=int, help="eval seed")
# tracking args
#main
parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
# parser.add_argument("--reid_path", type=str, default='./pretrained/lwtgpf.pt')
parser.add_argument("--min_init", type=int, default=3, help="oc-sort hit track")
parser.add_argument("--max_dist", type=float, default=0.2,help='max nn_match_distance')
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--dataset", type=str, default='mot17',help='Re-ID train dataset')
parser.add_argument("--test_dataset", type=bool, default=False,help='torchreid or fastreid')
parser.add_argument("--cmc_off", type=bool, default=False,help='cmc compensation')
#motion args
parser.add_argument("--motion_off", type=bool, default=False,help='motion metric')
parser.add_argument("--oc_match_thresh", type=float, default=0.2, help="oc-sort for origin iou matrix")
parser.add_argument("--delta_t", type=int, default=3, help="oc-sort time interval")
parser.add_argument("--inertia", type=float, default=0.2,help='weight of speed for ocsort')
#reid args
parser.add_argument("--embed_off", type=bool, default=False,help='embed metric')
parser.add_argument("--nn_budget", type=int, default=100,help='max number of save features')
parser.add_argument("--w_assoc_emb", type=float, default=0.75,help='base union appear weight')
parser.add_argument("--aw_parm", type=float, default=0.5,help='appear boost diff')
parser.add_argument("--grid-off", type=bool, default=True,help='horizontal split patches')
parser.add_argument("--dist_type", type=str, default='cosin',help='calculate appearance similarity attn/cosin')
parser.add_argument("--appear_thresh", type=float, default=0.3, help="ma-sort for appearance matrix")
parser.add_argument("--alpha", type=float, default=0.95,help='feature memory')
#Union args
parser.add_argument("--union_off", type=bool, default=False,help='union adaptive')
parser.add_argument("--alpha_gate", type=float, default=0.3,help='weight motion or appearance')
parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking iou-based")
#output args
parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

#HOTA参数
def get_hota_params(data_type='mot17'):
    EVAL_CONFIG= {'PRINT_RESULTS':False,'RETURN_ON_ERROR':True,'DISPLAY_LESS_PROGRESS':False}
    METRCIS_CONFIG = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    if data_type=='mot17':
        DATASET_CONFIG = {'GT_FOLDER':'./datasets/mot/',
                                'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                                'TRACKERS_TO_EVAL': None, #如果为None则会os.listdir
                                'TRACKER_SUB_FOLDER': 'data',
                                'BENCHMARK':'MOT17',
                                'SPLIT_TO_EVAL':'train',
                                'SKIP_SPLIT_FOL':False,
                                'GT_LOC_FORMAT':'{gt_folder}/{seq}/gt/gt_val_half.txt'}
    elif data_type=='mot20':
        DATASET_CONFIG = {'GT_FOLDER': './datasets/MOT20/',
                          'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                          'TRACKERS_TO_EVAL': None,  # 如果为None则会os.listdir
                          'TRACKER_SUB_FOLDER': 'data',
                          'BENCHMARK': 'MOT20',
                          'SPLIT_TO_EVAL': 'train',
                          'SKIP_SPLIT_FOL': False,
                          'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt_val_half.txt'}
    elif data_type=='dance':
        DATASET_CONFIG = {'GT_FOLDER': './datasets/dancetrack/',
                          'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                          'TRACKERS_TO_EVAL': None,  # 如果为None则会os.listdir
                          'TRACKER_SUB_FOLDER': 'data',
                          'BENCHMARK': 'Dancetrack',
                          'SPLIT_TO_EVAL': 'val',
                          'SKIP_SPLIT_FOL': False,
                          'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt'}
    else:
        raise ValueError(f'{data_type} is not support!')
    return EVAL_CONFIG,METRCIS_CONFIG,DATASET_CONFIG


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info(f"{emoji.emojize(':rocket:')} Computing {k}...")
            #todo: iou阈值设置为0.5
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning(f'{emoji.emojize(":warning:")} No ground truth for {k}, skipping.')

    return accs, names


def hota_preprocess(dataset_config,gtfiles,tsfiles):
    # step1：创建一个seqmaps存放seq名字
    seq_root = os.path.join(dataset_config['GT_FOLDER'], 'seqmaps')
    os.makedirs(seq_root, exist_ok=True)
    root_name = dataset_config['BENCHMARK'] + '-' + dataset_config['SPLIT_TO_EVAL']
    seq_names = ['name'] + [Path(x).parts[-3] for x in gtfiles]
    with open(os.path.join(seq_root, f'{root_name}.txt'), 'w') as f:
        for item in seq_names:
            f.write(f'{item}\n')
    logger.info(f"{emoji.emojize(':antenna_bars:')} \t Generate a seqmaps --> {os.path.join(seq_root, f'{root_name}.txt')} ")
    # step2: 创建tracker的文件夹
    split_root = Path(tsfiles[0]).parts
    ago_name = split_root[-2]
    track_fol = os.path.join(dataset_config['TRACKERS_FOLDER'], root_name, ago_name,
                             dataset_config['TRACKER_SUB_FOLDER'])
    os.makedirs(track_fol, exist_ok=True)
    _ = [shutil.copy(f, os.path.join(track_fol, os.path.basename(f))) for f in tsfiles if os.path.isfile(f)]
    logger.info(
        f"{emoji.emojize(':antenna_bars:')} \t Transfer track result from {os.path.dirname(tsfiles[0])} --> {track_fol} ")
    # step3:将GT的结果移到指定的文件夹下
    os.rename(os.path.join(dataset_config['GT_FOLDER'], dataset_config['SPLIT_TO_EVAL']), os.path.join(dataset_config['GT_FOLDER'], root_name))
    logger.info(f"{emoji.emojize(':antenna_bars:')} \t Rename GT folder from train --> {root_name} ")




def main():
    args = parser.parse_args()
    exp = get_exp(args.exp_file,None)
    exp.output_dir = args.out_root
    num_gpu = torch.cuda.device_count() if args.devices is not None else args.devices
    assert num_gpu<=torch.cuda.device_count()

    #Pre setting for env
    file_name = osp.join(exp.output_dir,exp.exp_name)
    os.makedirs(file_name,exist_ok=True)
    results_folder = osp.join(file_name,'track_results')
    os.makedirs(results_folder,exist_ok=True)
    setup_logger(file_name,distributed_rank=0,filename='val_log.txt',mode='a')
    logger.info(f"{emoji.emojize(':locked:')} Args: {args}")
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    exp.test_size = (args.tsize, args.tsize) if args.tsize is not None else exp.test_size

    #Instantiate Model,evaluator and dataloader
    model = exp.get_model()
    val_loader = exp.get_eval_loader(args.batch_size,is_distributed=False,testdev=args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size= exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )
    torch.cuda.set_device(0)
    model.cuda(0)
    model.eval()
    logger.info(f"{emoji.emojize(':grinning_face_with_big_eyes:')} Model Summary: {get_model_info(model, exp.test_size)}")
    logger.info(f"{emoji.emojize(':envelope:')} Dataloader and evaluator has been defined")
    #Load ckpt
    logger.info(f'{emoji.emojize(":rocket:")} loading checkpoint')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt = ckpt['model'] if 'model' in ckpt.keys() else ckpt
    misskeys, unexceptkeys = model.load_state_dict(ckpt, strict=False)
    logger.info(f"{emoji.emojize(':warning:')} missing: {misskeys}")
    logger.info(f"{emoji.emojize(':prohibited:')} except: {unexceptkeys}")

    #start evaluator
    if len(os.listdir(results_folder))==0:
        *_,summary = evaluator.evaluate(model,test_size=exp.test_size,result_folder=results_folder)
        logger.info('\n'+summary)

    #evaluate MOTA
    mm.lap.default_solver = 'lap'
    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    if args.dataset=='mot20':
        gtfiles = glob(osp.join('./datasets/MOT20/train',f'*/gt/gt{gt_type}.txt'))
    elif args.dataset=='mot17':
        gtfiles = glob(osp.join('./datasets/mot/train', f'*/gt/gt{gt_type}.txt'))
    elif args.dataset=='dance':
        gtfiles = glob(osp.join('./datasets/dancetrack/val', f'*/gt/gt.txt'))
    else:
        raise ValueError(f'{args.dataset} is not found!')
    tsfiles = [f for f in glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]
    logger.info(f"{emoji.emojize(':magnifying_glass_tilted_right:')} Found {len(gtfiles)} GTs and {len(tsfiles)} Preds")
    logger.info(f"{emoji.emojize(':hammer:')} Available LAP solver {mm.lap.available_solvers}")
    logger.info(f"{emoji.emojize(':hammer:')} Default LAP solver {mm.lap.available_solvers}")
    logger.info(f"{emoji.emojize(':envelope:'*5)} Loading files {emoji.emojize(':envelope:'*5)}")

    gt = OrderedDict([(Path(f).parts[-3],mm.io.loadtxt(f,fmt='mot15-2D',min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(Path(f).parts[-1].split('.')[0],mm.io.loadtxt(f,fmt='mot15-2D',min_confidence=-1)) for f in tsfiles])
    mh = mm.metrics.create()
    accs,names = compare_dataframes(gt,ts)
    logger.info(f"{emoji.emojize(':alien:'*3)} Running metrics {emoji.emojize(':alien:'*3)}")
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    logger.info(f"{emoji.emojize(':dog_face:')} Percentage version format:")
    logger.info("\n" + mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    logger.info(f"{emoji.emojize(':dog_face:')} Original version format:")
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    logger.info("\n" + mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info(f"{emoji.emojize(':dog_face:')}: Evaluate HOTA metrics...")
    EVAL_CONFIG,METRCIS_CONFIG,DATASET_CONFIG = get_hota_params(args.dataset)
    root_name = DATASET_CONFIG['BENCHMARK']+'-'+DATASET_CONFIG['SPLIT_TO_EVAL']
    hota_preprocess(DATASET_CONFIG,gtfiles,tsfiles)
    evaluator = trackeval.Evaluator(EVAL_CONFIG)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(DATASET_CONFIG)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in METRCIS_CONFIG['METRICS']:
            metrics_list.append(metric(METRCIS_CONFIG))
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    ago_name = Path(tsfiles[0]).parts[-2]
    output = output_res['MotChallenge2DBox'][ago_name]
    key_metrics_hota = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr']
    one_metric = {}
    for key, value in output.items():
        hot_rel = value['pedestrian']['HOTA']
        for m in key_metrics_hota:
            if m in hot_rel.keys():
                one_metric.setdefault(key, {}).setdefault(m, hot_rel[m].mean())
    df = pd.DataFrame(one_metric).transpose()
    df.index = df.index.map(lambda x: 'OVERALL' if x == 'COMBINED_SEQ' else x)
    df_hota = pd.merge(df, summary, how='right', left_index=True, right_index=True)
    hota_fmt = mh.formatters.copy()
    hota_metric = mm.io.motchallenge_metric_names.copy()
    update_fmt_dict = {}
    for k in key_metrics_hota:
        hota_fmt.setdefault(k.lower(), hota_fmt['mota'])
        hota_metric.setdefault(k.lower(), k)
    logger.info("\n"+mm.io.render_summary(df_hota, formatters=hota_fmt, namemap=hota_metric))
    #将gt的文件夹名称回到原来的名称
    os.rename(os.path.join(DATASET_CONFIG['GT_FOLDER'], root_name), os.path.join(DATASET_CONFIG['GT_FOLDER'], DATASET_CONFIG['SPLIT_TO_EVAL']))


if __name__ == '__main__':
    main()

