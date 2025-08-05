python tools/matrack.py --exp-file ./exps/example/mot/yolox_x_ablation.py \
-o ./masort/ablation -c ./pretrained/masortweight/masort_ablation.pth.tar \
--dataset mot17 --w_assoc_emb 0.75 --appear_thresh 0.3 --alpha_gate 0.3