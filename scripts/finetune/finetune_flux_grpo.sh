export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

mkdir -p images
export HCCL_BUFFSIZE=800

sudo apt-get update
yes | sudo apt-get install python3-tk

git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e . 
cd ..


###Actually, we don't use the original pytorch torchrun in our internal environment, 
###so I just follow the official example of pytorch.
###Please adapt the torchrun scripts into your own environment
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=192.168.0.101 --master_port=29500 \
    fastvideo/train_grpo_flux.py \
    --seed 42 \
    --pretrained_model_name_or_path data/flux \
    --vae_model_path data/flux \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 300 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 40 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \