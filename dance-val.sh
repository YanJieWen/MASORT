python tools/matrack.py --exp-file ./exps/example/dance/yolox_dancetrack_val.py \
-o ./masort/dance-val -c ./pretrained/masortweight/masort_dance.pth.tar \
--dataset dance  --w_assoc_emb 1.25--appear_thresh 0.2 --alpha_gate 0.9