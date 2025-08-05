python tools/matrack.py --exp-file ./exps/example/mot/yolox_x_mix_det.py \
-o ./masort/mot17 -t True -c ./pretrained/masortweight/masort_mot17.pth.tar \
--dataset mot17 --test_dataset True --w_assoc_emb 0.75 --appear_thresh 0.3 --alpha_gate 0.3