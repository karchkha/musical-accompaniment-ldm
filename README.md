# Towards Real-Time Musical Agents: Instrumental Accompaniment with Latent Diffusion Models and MAX/MSP



This repository contains the official PyTorch implementation accompanying the paper **"Towards Real-Time Musical Agents: Instrumental Accompaniment with Latent Diffusion Models and MAX/MSP"**.


- [arXiv](link here)  
- [Demo Page] [Demo page](https://consistency-separation.github.io/)

**Authors**: Tornike Karchkhadze, Shlomo Dubnov — University of California San Diego

---

## Abstract

We propose a framework for a real-time instrumental accompaniment and improvisation system. The project is twofold: we develop a diffusion-based generative model for musical accompaniment, and build a hybrid system that enables real-time interaction with this model by combining MAX/MSP with a remote Python server. Our latent diffusion model is trained with lookahead conditioning and deployed on a Python server. The MAX/MSP frontend handles real-time audio input, buffering, and playback, and communicates with the server via OSC messages. This setup enables a musician to plug in and play live within MAX/MSP, while the ML model listens and responds with complementary instrumental parts.

---

## System Overview
<p align="center">
  <img src="figures/Real_time_MAX.drawio.png" width="40%"/>
</p>

A human musician (e.g., drummer) performs live while a front-end computer running MAX/MSP captures the incoming audio stream and communicates with a remote GPU server via Open Sound Control (OSC). The server hosts a diffusion-based generative model that receives the audio input and generates accompaniment complementary instrument (e.g., bass) in real time. The generated audio is returned to the MAX/MSP environment and mixed with the human performance to produce the final musical output.

### Real-Time Sliding-Window Protocol

<p align="center">
  <img src="figures/Real_time_graph.drawio.png" width="95%"/>
</p>



Real-time accompaniment is formulated as a sliding-window generation process over a fixed-length context of duration *T*. The window advances by *T·r* at each step, where *r* controls the step size. Three regimes are supported: **retrospective** (w=−1), **immediate** (w=0), and **lookahead** (w=1) prediction.

### Latent Diffusion Model for Accompaniment

<p align="center">
  <img src="figures/latent.drawio.png" width="100%"/>
</p>

The accompaniment model encodes the input audio mixture into a latent representation via a pre-trained [Music2Latent](https://github.com/SonyCSLParis/music2latent) autoencoder, runs iterative denoising with a U-Net diffusion backbone (~257M parameters), and decodes the result back to audio. For real-time use the model can be run in **inpainting (lookahead) mode**, where partial context is provided as a condition.

### Consistency Distillation for Fast Inference

To meet real-time latency constraints, the diffusion model is distilled into a consistency model (CD). The student is trained to directly map noisy inputs to consistent estimates in 1–2 steps, guided by an EMA teacher and a combined consistency + DSM loss.

## Setup

### 1. Directory Structure

The repo expects two directories at its root:

| Path | Purpose |
|------|---------|
| `dataset/` | Slakh2100 dataset root |
| `lightning_logs/` | Training checkpoints and logs (written by PyTorch Lightning) |


### 2. Dataset

This project uses the [Slakh2100](http://www.slakh.com/) dataset processed at 44100 Hz (bass, drums, guitar, piano stems). A preparation script is provided that downloads the original Slakh2100 from Zenodo, groups stems by instrument class, creates mono WAV files and a mixture track, assigns train/validation/test splits by track number, and saves split metadata JSON files.

**Full dataset (~100 GB, 44100 Hz):**
```bash
python main/prepare_dataset/prepare_slakh.py \
    --splits train validation test \
    --dest dataset/slakh2100_44100
```

**Quick test with BabySlakh (~880 MB, 16 kHz — for verifying the pipeline only):**
```bash
python main/prepare_dataset/prepare_slakh.py \
    --splits train \
    --dest dataset/slakh2100_44100_tiny \
    --tiny
```

Tracks are assigned to splits by track number following the official Slakh2100 convention:

| Split | Track range | Count |
|---|---|---|
| train | Track00001 – Track01500 | ~1500 |
| validation | Track01501 – Track01875 | ~375 |
| test | Track01876 – Track02100 | ~225 |

### 3. Conda Environment

This repository uses Python 3.10.

**Linux (recommended):**
```bash
conda env create -f environment.yaml
conda activate ctm_gen
```

**Windows:**
```bash
conda env create -f environment_windows.yaml
conda activate ctm_gen
```

> On Windows, `flash-attn`, `xformers`, and `triton` are not available as pre-built wheels and are skipped. The code falls back to `torch.nn.functional.scaled_dot_product_attention` automatically.

**Mac (Apple Silicon):**
```bash
conda env create -f environment_mac_m.yaml
conda activate ctm_gen
```


**Post-install step (all platforms):** `audioldm_eval` must be installed manually and a specific veriosn. Run after activating the environment:

```bash
pip install --no-deps ssr-eval
pip install --no-deps "audioldm-eval @ git+https://github.com/haoheliu/audioldm_eval.git@8dc07ee7c42f9dc6e295460a1034175a0d49b436"
```

---

## Training

### Accompaniment Generation Models (LDM)

**Maskless diffusion model:**
```bash
python train_audio.py --cfg configs/generation/Diff_latent_cond_gen_concat_train.yaml
```

**Masked diffusion model (with lookahead support):**
```bash
python train_audio.py --cfg configs/generation/Diff_latent_cond_gen_concat_inpaint_train.yaml
```

**Maskless consistency distillation model:**
```bash
python main_audio_ctm.py --cfg configs/generation/CD/CD_latent_cond_gen_concat_train.yaml
```

**Masked consistency distillation model (with lookahead support):**
```bash
python main_audio_ctm.py --cfg configs/generation/CD/CD_latent_cond_gen_concat_inpaint_train.yaml
```
---

## Checkpoints

Checkpoints are available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19045462.svg)](https://doi.org/10.5281/zenodo.19045462)

Download and extract into the `lightning_logs/` directory:

```bash
cd lightning_logs
wget https://zenodo.org/record/19045462/files/checkpoints.tar.gz
tar -xzf checkpoints.tar.gz
rm checkpoints.tar.gz
```

The archive contains the two checkpoints used by the server configs:
- `GEN_diffusion_model/.../checkpoints/last.ckpt` — masked diffusion model
- `GEN_CD/.../checkpoints/last.ckpt` — masked consistency distillation model

---

## Evaluation

Evaluation is run via `main/eval/generate_eval.py`, which simulates the real-time sliding-window inpainting pipeline offline over the Slakh2100 test set and computes COCOLA, Beat F1, and FAD scores.

> The evaluation pipeline is adapted from [Streaming Generation for Music Accompaniment](https://github.com/lukewys/stream-music-gen).

### 1. Download Pretrained Evaluation Models

**COCOLA** (accompaniment quality metric):
```bash
mkdir -p lightning_logs/stream_music_gen/eval_models/cocola_models
gdown 1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ -O lightning_logs/stream_music_gen/eval_models/cocola_models/checkpoint-epoch=87-val_loss=0.00.ckpt
```

**Beat alignment** ([Beat This](https://github.com/CPJKU/beat_this)): weights are downloaded automatically on first run.

Alternatively, you can use [Beat Transformer](https://github.com/zhaojw1998/Beat-Transformer) by passing `--method beat_transformer`:
```bash
mkdir -p lightning_logs/stream_music_gen/eval_models/beat_transformer_models
wget -O lightning_logs/stream_music_gen/eval_models/beat_transformer_models/fold_4_trf_param.pt \
  https://github.com/zhaojw1998/Beat-Transformer/raw/main/checkpoint/fold_4_trf_param.pt
```

**FAD** (vggish / pann / clap / encodec): weights are downloaded automatically on first run by the `audioldm_eval` library.

### 2. Run Evaluation

```bash
python main/eval/generate_eval.py \
    --config configs/for_server/Diff_latent_cond_gen_concat_eval.yaml \
    --r 0.25 --w 0 \
    --num_samples 1024 \
    --device cuda:0 \
    --stems bass drums guitar piano
```

**Mac (Apple Silicon):**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main/eval/generate_eval.py \
    --config configs/for_server/Diff_latent_cond_gen_concat_eval.yaml \
    --r 0.25 --w 0 \
    --num_samples 1024 \
    --device mps \
    --stems bass drums guitar piano
```

---

## MAX/MSP Integration (Real-Time Server)

`server.py` (diffusion model) and `server_CD.py` (consistency distillation model) expose an OSC interface for real-time integration with MAX/MSP.

Form Max MAX/MSP we will be sending corespoing commands: 

**Run the diffusion server:**
```bash
python server.py --serverport 7000 --clientport 8000 --server_ip <YOUR_SERVER_IP>
```

**Run the CD server:**
```bash
python server_CD.py --serverport 7000 --clientport 8000 --server_ip <YOUR_SERVER_IP>
```

---

## Acknowledgments

This codebase builds upon the following repositories:

- [Sony CTM](https://github.com/sony/ctm)
- [Multi-Source Diffusion Models](https://github.com/gladia-research-group/multi-source-diffusion-models)
- [Audio Diffusion PyTorch (v0.43)](https://github.com/archinetai/audio-diffusion-pytorch)

---

## Citations

If you use this work, please cite:

```bibtex
```
