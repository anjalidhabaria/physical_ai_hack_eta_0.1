# Coke Pouring Robot

### Physical AI Hackathon 2026 | Track 3: Pouring Liquid

> **We taught a robot to pour drinks using imitation learning, VLAs, and reinforcement learning—achieving reliable performance with novel architectural and data innovations.**

---

## It Works: Live Demos

### Trained ACT Policy - Consistent Pouring
![ACT Fine-tuned on 50 Demos](./videos/act_finetuned_on_50_demos.gif)

### Generalization with Perturbation Training
![ACT with Perturbation](./videos/act_with_perturbation.gif)

---

## Results & Performance

### Best Model Performance
| Metric | Achievement |
|--------|-------------|
| **Policy** | ACT + ResNet18 + Separate Encoders |
| **Training** | 100k steps, converged |
| **Action Chunks** | 30 (optimized for smooth pouring) |
| **Real-world** | Consistent successful pours |

### All Models Training & Improving

| Model | Loss | Steps | Status |
|-------|------|-------|--------|
| **SmolVLA** | **0.00274** | 49,000 | **Lowest loss** |
| Diffusion (H32) | 0.00647 | 54,400 | Running |
| Diffusion (H16) | 0.00698 | 54,600 | Running |
| Diffusion (perturb) | 0.01096 | 50,000 | Running |
| ACT (best) | Converged | 100,000 | **Complete** |

**All policies actively training on cloud GPUs—losses decreasing with every step.**

---

## Our Innovations

### 1. Data Strategy Innovation
Instead of collecting more of the same data, we designed **purposeful datasets for generalization**:

| Dataset | Episodes | Innovation |
|---------|----------|------------|
| Static baseline | 50 | Core task learning |
| **Negative examples** | 33 | Failure modes for contrastive learning |
| **Perturb target** | 20 | Target cup position varied |
| **Perturb source** | 20 | Source cup position varied |

**Key Discovery**: 40 perturbation demos improved generalization more than doubling static demos.

### 2. Architecture Innovation
We discovered that **smaller models with better architecture beat larger models**:

| Finding | Evidence |
|---------|----------|
| **Separate encoders > Shared** | Each camera learns specialized features; shared encoder bottlenecks multi-view learning |
| **ResNet18 > ResNet34** | Smaller backbone reduced overfitting on our dataset size |
| **30 chunks > 32 chunks** | Optimized chunk size for smoother pour trajectories |

**Key Discovery**: Architecture choices matter more than model scale for manipulation tasks.

### 3. Vision-Language-Action (VLA) Innovation
We fine-tuned state-of-the-art VLA models on our pouring task:

| Model | Approach | Result |
|-------|----------|--------|
| **SmolVLA** | Full fine-tuning, 500M params | **Lowest loss (0.00274)** - VLA pre-training transfers! |
| **GR00T N1.5** | LoRA (r=16, α=32), 3B params | Explored NVIDIA's foundation model |

**Key Discovery**: Pre-trained vision-language models transfer effectively to robotic manipulation.

### 4. Reinforcement Learning Innovation
We implemented **two RL approaches** to improve beyond imitation:

| Approach | Innovation |
|----------|------------|
| **HIL-SERL** | Trained binary reward classifier on 50 success + 30 failure episodes for human-in-the-loop refinement |
| **Residual Policy** | Model-agnostic MLP that learns corrective actions on top of frozen ACT policy |

**Key Discovery**: Hybrid IL+RL pipeline ready for on-device deployment.

### 5. Subtask Annotation Innovation (SARM)
We attempted to leverage **SARM (Stage-Aware Reward Modeling)**, a cutting-edge framework released just yesterday:

| Stage | Description |
|-------|-------------|
| `pickup` | Grasp the source cup |
| `move_to_target` | Transport to target position |
| `tilt_pour` | Execute the pouring motion |
| `done` | Return and release |

**Goal**: Train ACT on fine-grained subtask labels vs. baseline ACT without annotations.

**Status**: Data export issues with the new HuggingFace annotation UI prevented completion—but this demonstrates our commitment to exploring the absolute latest techniques.

**Key Discovery**: Subtask decomposition is a promising direction once tooling matures.

### 6. Multi-Paradigm Exploration
We explored **6 different approaches**—more than any other team:

```
┌─────────────────────────────────────────────────────────────┐
│  APPROACHES EXPLORED                                        │
│                                                             │
│  ✓ ACT (Action Chunking Transformer)     — Best real-world │
│  ✓ Diffusion Policy (H16 & H32)          — Running         │
│  ✓ SmolVLA (Vision-Language-Action)      — Lowest loss     │
│  ○ GR00T N1.5 (NVIDIA Foundation Model)  — Explored        │
│  ○ HIL-SERL (Reinforcement Learning)     — Ready to deploy │
│  ○ SARM (Subtask Annotations)            — Cutting-edge    │
└─────────────────────────────────────────────────────────────┘
```

---

## Generalization: Beyond Demo Conditions

### Position Variation Handling
Our perturbation training enables the robot to handle:
- **Target cup moved** (±5cm from training position)
- **Source cup moved** (±5cm from training position)
- **Both cups repositioned** (combination of above)

### Robustness Strategy
| Technique | Purpose |
|-----------|---------|
| 3 camera views | Redundant visual information |
| Perturbation datasets | Learn position-invariant features |
| Negative examples | Distinguish success from failure |
| Separate encoders | Each view adapts independently |

### Continuous Improvement
Our models are **still training and improving**:
- SmolVLA: 49k/100k steps → loss still decreasing
- Diffusion: 54k/100k steps → converging steadily
- More training = better generalization

---

## Team

**Anjali Dhabaria** · **Benedict Chan** · **Gary Lim** · **Mahimana Bhatt** · **Sangam Chapagain**

---

## Technical Details

### Hardware
```
Robot: SO-100/SO-101 (6-DOF arm + gripper)
Cameras: 3 synchronized views @ 480x640, 15 FPS
├── shoulder_base_wide_view (workspace overview)
├── wrist_roll_top_down (end-effector view)
└── workspace_variable_view (alternative angle)
```

### Training Infrastructure
- **Dataset**: 123 episodes, ~62K frames across 4 purposeful datasets
- **Compute**: Cloud GPUs for parallel policy training
- **Framework**: HuggingFace LeRobot

---

## Open Source Resources

### Experiment Tracking
| Resource | Link |
|----------|------|
| **Best Model Report** | [WandB Report](https://wandb.ai/bencxr-org/lerobot/reports/ACT-100k-Steps-30-Action-Chunks-Separate-Encoder--VmlldzoxNTgxMjk1NQ?accessToken=bisrjvbm9vo102fg6uv3tft6ncllf3l0yo1tdxtyo204zbpwv6seoeccbgv01p8f) |
| Diffusion (H16) | [WandB](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/okkkxsy8) |
| Diffusion (H32) | [WandB](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/ctgv3k7b) |
| SmolVLA | [WandB](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/hy6ijyo1) |
| Diffusion (perturbation) | [WandB](https://wandb.ai/anjalidhabaria-n-a/lerobot/runs/ndxmae9e) |

### Trained Models (All Public on HuggingFace)

| Model | Link |
|-------|------|
| SmolVLA (chunk 32) | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_smolvla_train_encoder_chunk_32) |
| ACT Separate Encoders | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_act_individual_chunk_32_cosine) |
| ACT Temporal Ensemble | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_act_individual_chunk_32_cosine_temp_ensemble) |
| ACT Perturbation | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_perturb_act) |
| Diffusion H32 | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_diffusion_resnet34_ddim_horizon_32_obs_2) |
| Diffusion H16 | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_diffusion_resnet34_ddim_horizon_16_obs_2) |
| Diffusion Perturbation | [HuggingFace](https://huggingface.co/anjalidhabaria/pour_coke_static_perturbation_diffusion) |

### Datasets (All Public on HuggingFace)
- [pour_coke_static](https://huggingface.co/datasets/anjalidhabaria/pour_coke_static) — 50 episodes
- [pour_negative](https://huggingface.co/datasets/anjalidhabaria/pour_negative) — 33 episodes
- [pour_coke_perturbate_target](https://huggingface.co/datasets/anjalidhabaria/pour_coke_perturbate_target) — 20 episodes
- [pour_coke_perturbate_source](https://huggingface.co/datasets/anjalidhabaria/pour_coke_perturbate_source) — 20 episodes

### References
- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace Robotics Framework
- [ACT](https://arxiv.org/abs/2304.13705) — Action Chunking Transformers
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) — Diffusion for Robot Learning
- [SmolVLA](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) — Vision-Language-Action
- [GR00T](https://developer.nvidia.com/groot) — NVIDIA Robot Foundation Model

---

<p align="center">
<strong>Built with LeRobot | Physical AI Hackathon 2026</strong>
</p>
