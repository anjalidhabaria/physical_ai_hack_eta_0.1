# Physical AI Hackathon 2026 - Coke Pouring Robot

## Team Members
- Anjali Dhabaria
- Benedict Chan
- Gary Lim
- Mahimana Bhatt
- Sangam Chapagain

---

## Project Overview

**Task**: Track 3 - Pouring Liquid
**Goal**: Pour 200ml from a source container (black cup) into a target container (white cup)
**Robot**: SO-100 / SO-101 follower arm

### Competition Metrics
| Metric | Target |
|--------|--------|
| Volume Accuracy | >95% (±10ml tolerance) |
| Spillage | <5ml |
| Attempts | 3-10 declared attempts |

---

## Hardware Setup

- **Robot**: SO-100/SO-101 6-DOF arm with gripper
- **Action Space**: 6 joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **Cameras**: 3 synchronized views at 480x640, 15 FPS, AV1 codec
  - `shoulder_base_wide_view` - Wide workspace view
  - `wrist_roll_top_down` - End-effector perspective
  - `workspace_variable_view` - Alternative angle
- **Workspace**: Bounded area with yellow fabric barriers

---

## Data Collection

### Datasets (Total: 123 episodes, ~62K frames)

| Dataset | Episodes | Frames | Purpose |
|---------|----------|--------|---------|
| [pour_coke_static](https://huggingface.co/datasets/anjalidhabaria/pour_coke_static) | 50 | 31,421 | Main training - fixed cup positions |
| [pour_negative](https://huggingface.co/datasets/anjalidhabaria/pour_negative) | 33 | 8,179 | Negative/failure examples |
| [pour_coke_perturbate_target](https://huggingface.co/datasets/anjalidhabaria/pour_coke_perturbate_target) | 20 | 10,692 | Target (white) cup position varied |
| [pour_coke_perturbate_source](https://huggingface.co/datasets/anjalidhabaria/pour_coke_perturbate_source) | 20 | 11,531 | Source (black) cup position varied |

### Data Strategy
- **Static baseline**: Core task learning with fixed positions
- **Perturbations**: Improve generalization to position changes
- **Negative examples**: Learn failure modes for potential contrastive learning

---

## Experiments

### Imitation Learning

#### ACT (Action Chunking Transformer)

| Run | Backbone | Encoder | Chunk Size | Action Steps | Scheduler | Notes |
|-----|----------|---------|------------|--------------|-----------|-------|
| 1 | ResNet18 | Shared | 32 | 32 | None | Baseline |
| 2 | ResNet34 | Separate | 32 | 32 | Cosine + warmup | Enhanced backbone |
| 3 | ResNet34 | Separate | 32 | 1 | Cosine + warmup | Temporal ensemble (coeff=0.01) |
| **4** | **ResNet18** | **Separate** | **30** | **30** | Cosine + warmup | **Best performing** |

**Training Commands**:
```bash
# Training 1 - ACT Baseline
lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=act \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  --policy.use_separate_encoder_per_camera=false \
  --policy.vision_backbone=resnet18 \
  --policy.use_vae=true \
  --batch_size=32 \
  --steps=100000

# Training 2 - ACT with Separate Encoders (Best Performing)
lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=act \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  --policy.use_separate_encoder_per_camera=true \
  --policy.vision_backbone=resnet34 \
  --policy.use_vae=true \
  --batch_size=32 \
  --scheduler.type=diffuser \
  --scheduler.name=cosine \
  --scheduler.num_warmup_steps=500 \
  --steps=100000

# Training 3 - ACT with Temporal Ensemble
lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=act \
  --policy.chunk_size=32 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.use_separate_encoder_per_camera=true \
  --policy.vision_backbone=resnet34 \
  --policy.use_vae=true \
  --batch_size=32 \
  --steps=100000
```

#### Diffusion Policy

| Run | Backbone | Noise Scheduler | Horizon | Action Steps | Obs Steps |
|-----|----------|-----------------|---------|--------------|-----------|
| 4 | ResNet34 | DDIM | 16 | 8 | 2 |
| 5 | ResNet34 | DDIM | 32 | 16 | 2 |

**Training Commands**:
```bash
# Training 4 - Diffusion (Horizon 16)
lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=diffusion \
  --policy.n_obs_steps=2 \
  --policy.noise_scheduler_type=DDIM \
  --policy.vision_backbone=resnet34 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --batch_size=32 \
  --steps=100000

# Training 5 - Diffusion (Horizon 32)
lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=diffusion \
  --policy.n_obs_steps=2 \
  --policy.noise_scheduler_type=DDIM \
  --policy.vision_backbone=resnet34 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.horizon=32 \
  --policy.n_action_steps=16 \
  --batch_size=32 \
  --steps=100000
```

### Vision-Language-Action (VLA) Models

| Run | Model | Base | Training Mode | Chunk Size |
|-----|-------|------|---------------|------------|
| 6 | SmolVLA | SmolVLM2-500M-Video-Instruct | Full training (encoder unfrozen) | 32 |
| 7 | GR00T N1.5 | nvidia/GR00T-N1.5-3B | LoRA (r=16, α=32) + visual/projector/diffusion | 32 |

**Training Commands**:
```bash
# Training 6 - SmolVLA
lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  --batch_size=32 \
  --steps=100000

# Training 7 - GR00T N1.5
TRANSFORMERS_FIX_MISTRAL_REGEX=1 lerobot-train \
  --dataset.repo_id=anjalidhabaria/pour_coke_static \
  --policy.type=groot \
  --policy.tune_llm=false \
  --policy.tune_visual=true \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.n_action_steps=32 \
  --policy.chunk_size=32 \
  --policy.lora_rank=16 \
  --policy.lora_alpha=32 \
  --policy.lora_dropout=0.1 \
  --batch_size=16 \
  --steps=100000
```

### Reinforcement Learning

#### HIL-SERL (Human-in-the-Loop Sample-Efficient RL)
- **Data**: 50 success episodes + 30 failure episodes
- **Approach**: Binary reward classifier for success/failure
- **Features**: Human interventions to guide policy training
- **Status**: Not trained on-device due to time constraints

#### Residual Policy RL
- **Approach**: Model-agnostic, builds on frozen ACT policy
- **Architecture**: MLP producing corrective residual actions
- **Rewards**: Sparse human-provided rewards

---

## Results & Findings

### Best Performing Model
**ACT with Separate Image Encoders + ResNet18** (Training Run 4)
- ResNet18 backbone (smaller, faster)
- Separate encoder per camera
- 30 action chunks
- 100k training steps
- Cosine learning rate scheduler

[View Full WandB Report](https://wandb.ai/bencxr-org/lerobot/reports/ACT-100k-Steps-30-Action-Chunks-Separate-Encoder--VmlldzoxNTgxMjk1NQ?accessToken=bisrjvbm9vo102fg6uv3tft6ncllf3l0yo1tdxtyo204zbpwv6seoeccbgv01p8f)

**Key Finding**: Smaller ResNet18 backbone with separate encoders outperformed larger ResNet34, suggesting that model capacity isn't the bottleneck for this task.

### Key Observations

| Finding | Details |
|---------|---------|
| **Separate encoders help** | Individual encoders per camera outperformed shared encoder |
| **Generalization is the bottleneck** | Policy works well on static positions but struggles with cup position changes |
| **Perturbation data shows promise** | At 10k steps, some generalization observed but not robust |
| **More training needed** | 10k steps insufficient; target 100k steps for better convergence |

### Challenges

1. **Position sensitivity**: Small cup position changes require large action adjustments
2. **Tilt angle precision**: Pouring angle highly sensitive to cup-to-cup distance
3. **Depth perception**: Single viewpoint may not capture 3D relationships well

---

## WandB Experiment Tracking

### Training Metrics

| Run ID | Experiment | Final Loss | Steps | Status |
|--------|------------|------------|-------|--------|
| [okkkxsy8](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/okkkxsy8) | Diffusion (horizon 16) | **0.006982** | 54,600 | running |
| [ctgv3k7b](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/ctgv3k7b) | Diffusion (horizon 32) | **0.006473** | 54,400 | running |
| [hy6ijyo1](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/hy6ijyo1) | SmolVLA | **0.002736** | 49,000 | running |
| [wffzeoek](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/wffzeoek) | GR00T (LoRA) | N/A | N/A | failed |
| [ndxmae9e](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/ndxmae9e) | Diffusion (perturbation) | **0.010956** | 50,000 | running |

### Loss Analysis

1. **SmolVLA achieves lowest loss** (0.00274) - VLA pre-training provides strong initialization
2. **Diffusion horizon 32 outperforms horizon 16** (0.00647 vs 0.00698) - longer action horizons benefit pouring
3. **GR00T training failed** - requires debugging (possible memory or config issue)
4. **Perturbation data has higher loss** (0.01096) - expected for harder generalization task

---

## Key Insights

1. **Architecture matters**: Separate encoders allow each camera view to learn specialized features
2. **Smaller can be better**: ResNet18 with separate encoders outperformed ResNet34 - model capacity isn't the bottleneck
3. **Data diversity is critical**: Static demonstrations alone don't generalize; perturbations help
4. **Training duration**: Early stopping at 10k steps shows promise but needs longer training (100k optimal)
5. **VLA models show promise**: SmolVLA achieved lowest loss (0.00274), suggesting pre-trained vision-language models transfer well to manipulation
6. **Longer horizons help**: Diffusion with horizon 32 outperformed horizon 16 for the pouring task
7. **Perturbation adds difficulty**: Higher loss on perturbation data (0.011 vs 0.006) indicates the model is learning harder variations

---

## Next Steps

### Immediate (with more time)
- [ ] Train combined dataset (static + perturbations) for 100k steps
- [ ] Enable image augmentations (`--dataset.image_transforms.enable=true`)
- [ ] Deploy HIL-SERL on-device for online refinement

### Future Work
- [ ] Add depth camera for better 3D perception
- [ ] Implement liquid level detection for pour termination
- [ ] Explore sim-to-real with pouring simulation
- [ ] Fine-tune VLA models on combined dataset

---

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Physical AI Hackathon Rules](https://physicalaihack.com/rules)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)
- [SmolVLA](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- [GR00T N1.5](https://developer.nvidia.com/groot)

---

*Report generated for Physical AI Hackathon 2026*
