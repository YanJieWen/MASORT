python tools/matrack.py --exp-file ./exps/example/dance/yolox_x_dancetrack_test.py \
-o ./masort/dance-test -t True -c ./pretrained/masortweight/masort_dance.pth.tar \
--dataset dance --test_dataset True --w_assoc_emb 1.25 --appear_thresh 0.2 --alpha_gate 0.9