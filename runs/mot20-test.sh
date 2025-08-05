python tools/matrack.py --exp-file ./exps/example/mot/yolox_x_mix_mot20_ch.py \
-o ./masort/mot20 -t True -c ./pretrained/masortweight/masort_mot20.tar \
--dataset mot20 --test_dataset True --w_assoc_emb 0.75 --appear_thresh 0.3 --alpha_gate 0