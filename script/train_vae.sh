if [ -z "$1" ]
    then
    echo "Require NGPU input; "
    exit
fi
DATA=" ddpm.input_dim 3 data.cates c1 "
NGPU=$1 # 
num_node=1
BS=2
total_bs=$(( $NGPU * $BS ))
if (( $total_bs > 128 )); then 
    echo "[WARNING] total batch_size larger than 128 may lead to unstable training, please reduce the size"
    exit
fi

# Choose training mode based on GPU count
if [ $NGPU -eq 1 ]; then
    echo "Using single GPU training"
    ENT="torchrun --nproc-per-node=$NGPU train_dist.py "
    NUM_WORKERS=4
elif [ $NGPU -le 4 ]; then
    echo "Using DistributedDataParallel for $NGPU GPUs"
    ENT="torchrun --nproc-per-node=$NGPU train_dist.py "
    NUM_WORKERS=2
else
    echo "Using DistributedDataParallel for $NGPU GPUs"
    ENT="torchrun --nproc-per-node=$NGPU train_dist.py "
    NUM_WORKERS=2
    # For DDP with 8 GPUs, each GPU gets its own process and can handle larger batch sizes
    # Each GPU should get at least 2-4 samples for efficient processing
    BS=$(( $BS * 2 ))  # 2 * 2 = 4, each GPU process gets 4 samples
    echo "Set batch size to $BS per GPU for DDP - total batch size: $(( $BS * $NGPU ))"
    # Explicitly set all GPUs to be visible
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

kl=0.5  
lr=1e-3
latent=1
skip_weight=0.01 
sigma_offset=6.0
loss='l1_sum'

echo "Using $NUM_WORKERS data workers for $NGPU GPUs"

# Set NCCL environment variables for better stability (only used in distributed mode)
if [ $NGPU -le 4 ] && [ $NGPU -gt 1 ]; then
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=WARN
    export NCCL_TIMEOUT=1800
    export NCCL_HEARTBEAT_TIMEOUT_SEC=300
fi

$ENT ddpm.num_steps 1 ddpm.ema 0 \
    trainer.opt.vae_lr_warmup_epochs 0 \
    latent_pts.ada_mlp_init_scale 0.1 \
    sde.kl_const_coeff_vada 1e-7 \
    trainer.anneal_kl 1 sde.kl_max_coeff_vada $kl \
    sde.kl_anneal_portion_vada 0.5 \
    shapelatent.log_sigma_offset $sigma_offset latent_pts.skip_weight $skip_weight \
    trainer.opt.beta2 0.99 \
    data.num_workers $NUM_WORKERS \
    ddpm.loss_weight_emd 1.0 \
    trainer.epochs 8000 data.random_subsample 1 \
    viz.viz_freq -400 viz.log_freq -1 viz.val_freq 200 \
    data.batch_size $BS viz.save_freq 2000 \
    trainer.type 'trainers.hvae_trainer' \
    model_config default shapelatent.model 'models.vae_adain' \
    shapelatent.decoder_type 'models.latent_points_ada.LatentPointDecPVC' \
    shapelatent.encoder_type 'models.latent_points_ada.PointTransPVC' \
    latent_pts.style_encoder 'models.shapelatent_modules.PointNetPlusEncoder' \
    shapelatent.prior_type normal \
    shapelatent.latent_dim $latent trainer.opt.lr $lr \
    shapelatent.kl_weight ${kl} \
    shapelatent.decoder_num_points 2048 \
    data.tr_max_sample_points 2048 data.te_max_sample_points 2048 \
    ddpm.loss_type $loss cmt "lion" \
    $DATA viz.viz_order [2,0,1] data.recenter_per_shape False data.normalize_global True 
