python3 main.py \
    --gpus 0,1 \
    --result_path results/3 \
    --bash_path KETI.sh \
    --root_path /workspace/3D/pytorch/HandSign/ \
    --video_path KETI_jpg/ \
    --annotation_path datasets/KETI_SL_util/KETI.json \
    --dataset KETI    \
    --model resnetl  \
    --model_depth 10  \
    --store_name 10 \
    --n_classes 419 \
    --batch_size 128 \
    --n_threads  0 \
    --learning_rate 0.1 \
    --n_epochs 70 \
    --n_val_samples 1 \
    --sample_duration 64 \
    --train_crop center \
    --scale_step 0.89089641525
