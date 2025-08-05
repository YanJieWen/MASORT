# MASORT

## Abstract
Human-centric video data and algorithms have been driving advances in Multi-Object Tracking (MOT) community. Most popular MOT methods follow the Detection-Based Tracking (DBT) paradigm and achieve impressive performance.  Simple Online and Realtime Tracking (SORT) serves as a foundational association algorithm is continuously optimized in subsequent researches. However, DBT methods still face 2 challenges. On the one hand, when a target suffers from occlusion for prolonged time, Kalman filter (KF)-based motion model trusts the prior state estimations to perform pseudo-updates, leading to significant motion variance. On the other hand, original association paradigms assume behavioral homogeneity among pedestrians and apply the same features for all items. This uniform treatment hinders the advantages of different features can not be fully utilized. To this end, we propose a Measurement-assisted Adaptive association (MASORT) for tracking multi-pedestrian with behavior discrepancies. MASORT adopts the 3-stage association framework of ByteTrack. To fully exploit the effectiveness of motion and Re-ID features, we introduce a Measurement-assisted Motion Enhancement (M2E) module and an Appearance Enhancement (AE) module. M2E improves KF via measurements instead of estimations and proposes a momentum-consistent motion similarity metric. AE dynamically updates Re-ID embeddings based on confidence scores and emphasizes persons with discriminative appearance. Furthermore, we design an Appearance-Motion Adaptive Association (AMAA) algorithm to accommodate behavior discrepancies among individuals. AMAA selects a  suitable feature based on each person’s motion level. Extensive experiments on 3 MOT benchmarks, including MOT17, MOT20 and DanceTrack, demonstrate that MASORT achieves the optimal trade-off between detection and association performance. On the challenging DanceTrack dataset, where the object appearance is highly similar, our MASORT  attains a remarkable 61.4 HOTA. 
<p align="center"><img src="assets/masort.jpg" width="800"/></p> 

## News 

## Tracking performance
### Results on MOT challenge and DanceTrack test set

## Preliminary
### 1. Installing on your local machine ⌨
Step1. Install MASORT
```shell
git clone  https://github.com/YanJieWen/MASORT.git
pip install emoji
python setup.py develop
```
Step2. Install motmetrics
```shell
cd MASORT
pip install motmetrics
```

Step3. Install [TrackEval](https://github.com/JonathonLuiten/TrackEval)
```shell
cd external
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -v -e .
```

Step3. Install [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
```shell
cd external
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py develop
```

Step4. Install [Fastreid](https://github.com/JDAI-CV/fast-reid)
```shell
cd external
git clone https://github.com/JDAI-CV/fast-reid.git
```

## Data preparation🔥
| MOTChallenge | DanceTrack |
|:-----------------:|:----------------:|
|[![MOT](https://img.shields.io/badge/😈mot-blue)](https://motchallenge.net/)|[![Dance](https://img.shields.io/badge/😈dance-challenge-blue)](https://github.com/DanceTrack/DanceTrack)|

Then, you need to turn the datasets to COCO format and mix different training data:

```shell
cd MASORT
python tools/convert_mot17_to_coco.py
python tools/convert_mot20_to_coco.py
python tools/convert_crowdhuman_to_coco.py
python tools/convert_cityperson_to_coco.py
python tools/convert_ethz_to_coco.py
```
It is worth noting that for the MOT17 test set, we uploaded the ``JSON`` file of [FRCNN](datasets/mot/test-FRCNN.json) to accelerate the evaluation.  

Subsequently, we perform file management operations as instructed in [tools/mix_xxx](tools), and run the following script:
```shell
cd MASORT
python tools/mix_data_ablation.py
python tools/mix_data_test_mot17.py
python tools/mix_data_test_mot20.py
```


## Model ZOO

