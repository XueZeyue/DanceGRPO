## DanceGRPO
**DanceGRPO is the first unified RL-based framework for visual generation.**

This is the official implementation for [paper](https://arxiv.org/abs/2505.07818), DanceGRPO: Unleashing GRPO on Visual Generation.
We develop [DanceGRPO](https://arxiv.org/abs/2505.07818) based on FastVideo, a scalable and efficient framework for video and image generation.

## Key Features

DanceGRPO has the following features:
- Support Stable Diffusion
- Support FLUX
- Support HunyuanVideo

## Updates

- __[2025.05.12]__: 🔥 We released the paper in arXiv!
- __[2025.05.28]__: 🔥 We released the training scripts of FLUX and Stable Diffusion!
- __[2025.07.03]__: 🔥 We released the training scripts of HunyuanVideo!

We have shared this work at many research labs, and the example slide can be found [here](https://github.com/XueZeyue/xuezeyue.github.io/blob/main/_talks/dancegrpo.pdf).

## Getting Started
### Downloading checkpoints
You should use ```"mkdir"``` for these folders first. 

For image generation,
1. Download the Stable Diffusion v1.4 checkpoints from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) to ```"./data/stable-diffusion-v1-4"```.
2. Download the FLUX checkpoints from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev) to ```"./data/flux"```.
3. Download the HPS-v2.1 checkpoint (HPS_v2.1_compressed.pt) from [here](https://huggingface.co/xswu/HPSv2/tree/main) to ```"./hps_ckpt"```.
4. Download the CLIP H-14 checkpoint (open_clip_pytorch_model.bin) from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) to ```"./hps_ckpt"```.

For video generation,
1. Download the HunyuanVideo checkpoints from [here](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) to ```"./data/HunyuanVideo"```.
2. Download the Qwen2-VL-2B-Instruct checkpoints from [here](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) to ```"./Qwen2-VL-2B-Instruct"```.
3. Download the VideoAlign checkpoints from [here](https://huggingface.co/KwaiVGI/VideoReward) to ```"./videoalign_ckpt"```.

### Installation
```bash
./env_setup.sh fastvideo
```
If you are using NPU for training, you need to modify the following code:
```python
# your_path_to_python/site-packages/diffusers/models/embeddings.py line 1250
is mps = ids.device.type == "mps"
is_npu = ids.device.type == "npu" # modified
freqs_dtype = torch.float3 if is_mps or is _npu else torch.float64 # modified
```

### Training

```bash
# for Stable Diffusion, with 8 H800 GPUs
bash scripts/finetune/finetune_sd_grpo.sh   
```
```bash
# for FLUX, preprocessing with 8 H800 GPUs
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
# for FLUX, training with 16 H800 GPUs for better convergence,
# or you can use finetune_flux_grpo_8gpus.sh with 8 H800 GPUs, but with relatively slower convergence
bash scripts/finetune/finetune_flux_grpo.sh   
```

For image generation open-source version, we use the prompts in [HPD](https://huggingface.co/datasets/ymhao/HPDv2/tree/main) dataset for training, as shown in ```"./prompts.txt"```.

```bash
# for HunyuanVideo, preprocessing with 8 H800 GPUs
bash scripts/preprocess/preprocess_hunyuan_rl_embeddings.sh
# for HunyuanVideo, using the following script for training with 16/32 H800 GPUs
bash scripts/finetune/finetune_hunyuan_grpo.sh   
```

For the video generation open-source version, we filter the prompts from [VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM) dataset for training, as shown in ```"./video_prompts.txt"```.

### Image Generation Rewards
We give the (moving average) reward curves (also the results in **`reward.txt`** or **`hps_reward.txt`**) of Stable Diffusion (left or upper) and FLUX (right or lower). We can complete the FLUX training (200 iterations) within **12 hours** with 16 H800 GPUs.

<img src=assets/rewards/opensource_sd.png width="49%">
<img src=assets/rewards/opensource_flux.png width="49%">

1. We provide more visualization examples (base, 80 iters rlhf, 160 iters rlhf) in ```"./assets/flux_visualization"```. 
2. Here is the visualization script `"./scripts/visualization/vis_flux.py"` for FLUX. First, run `rm -rf ./data/flux/transformer/*` to clear the directory, then copy the files from a trained checkpoint (e.g., `checkpoint-160-0`) into `./data/flux/transformer`. After that, you can run the visualization. If it's trained for 160 iterations, the results are already provided in my repo.  
3. More discussion on FLUX can be found in ```"./fastvideo/README.md"```.
4. (Thanks for a community contribution from [@Jinfa Huang](https://infaaa.github.io/), if you change the train_batch_size and train_sp_batch_size from 1 to 2, change the gradient_accumulation_steps from 4 to 12, **you can train the FLUX with 8 H800 GPUs**, and you can finish the FLUX training within a day. If you experience a reward collapse similar to [this](https://github.com/XueZeyue/DanceGRPO/issues/55), please reduce the `max_grad_norm`.)


### Video Generation Rewards
We give the (moving average) reward curves (also the results in **`vq_reward.txt`**) of HunyuanVideo with 16/32 H800 GPUs.

With 16 H800 GPUs,

<img src=assets/rewards/opensource_hunyuanvideo_16gpus.png width="49%">

With 32 H800 GPUs,

<img src=assets/rewards/opensource_hunyuanvideo_32gpus.png width="49%">

1. For the open-source version, our mission is to reduce the training cost. So we reduce the number of frames, sampling steps, and GPUs compared with the settings in the paper. So the reward curves will be different, but the VQ improvements are similar (50%~60%). 
2. For visualization, run `rm -rf ./data/HunyuanVideo/transformer/*` to clear the directory, then copy the files from a trained checkpoint (e.g., `checkpoint-100-0`) into `./data/HunyuanVideo/transformer`. After that, you can run the visualization script `"./scripts/visualization/vis_hunyuanvideo.sh"`.
3. Although training with 16 H800 GPUs has similar rewards with 32 H800 GPUs, I still find that 32 H800 GPUs leads to better visulization results.
4. We plot the rewards by **de-normalizing**, with the formula VQ = VQ * 2.2476 + 3.6757 by following [here](https://huggingface.co/KwaiVGI/VideoReward/blob/main/model_config.json).


### Multi-reward Training
The Multi-reward training code and reward curves can be found [here](https://github.com/XueZeyue/DanceGRPO/issues/19).

### Important Discussion and Results with More Reward Models
Thanks for the issue from [@Yi-Xuan XU](https://github.com/xuyxu), the results of more reward models and better visualization (how to avoid grid patterns) on FLUX can be found [here](https://github.com/XueZeyue/DanceGRPO/issues/36). We also support the pickscore for FLUX with `--use_pickscore`.

We support the EMA for FLUX with `--ema_decay 0.995` and `--use_ema`. Enabling EMA helps with better visualization.

## How to Support Custom Models
1. For preprocessing, modify the `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` based on your text encoder.
2. For FSDP and dataloader, modify the `fsdp_util.py` and `communications_flux.py`, we prefer FSDP rather than DeepSpeed since FSDP is easier to debug.
3. Modify the `train_grpo_flux.py`.

How to debug:
1. Print the probability ratio, reward, and advantage for each sample; the ratio should be **1.0** before the gradient update, and you can verify the advantage on your own. **Please set the rollout inference batch size and training batch size to 1, otherwise you will not have the ratio 1.0.**
2. The gradient accumulation should follow the sample dimension, which means, suppose you use 20 steps, the gradient accumulation should be accumulate_samples*20.
3. Based on our experience, the learning rate should be set to between 5e-6 and 2e-5, setting the lr to 1e-6 always leads to training failure in our settings.
4. Make sure the batchsize is enough; you can follow our setting of flux_8gpus.
5. More importantly, if you enable cfg, the gradient accumulation should be set to a large number. Based on our experience, we always set it to be num_generations*20, which means you update the gradient only once in each rollout.


## Training Acceleration
1. You can reduce the sampling steps, resolution, or timestep selection ratio.
2. Thanks for the outstanding follow-up work [MixGRPO](https://arxiv.org/abs/2507.21802), please refer to [here](https://github.com/Tencent-Hunyuan/MixGRPO).


## Acknowledgement
We learned and reused code from the following projects:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [diffusers](https://github.com/huggingface/diffusers)
- [DDPO-Pytorch](https://github.com/kvablack/ddpo-pytorch)

We thank the authors for their contributions to the community!

## Citation
If you use DanceGRPO for your research, please cite our paper:

```bibtex
@article{xue2025dancegrpo,
  title={DanceGRPO: Unleashing GRPO on Visual Generation},
  author={Xue, Zeyue and Wu, Jie and Gao, Yu and Kong, Fangyuan and Zhu, Lingting and Chen, Mengzhao and Liu, Zhiheng and Liu, Wei and Guo, Qiushan and Huang, Weilin and others},
  journal={arXiv preprint arXiv:2505.07818},
  year={2025}
}
```
