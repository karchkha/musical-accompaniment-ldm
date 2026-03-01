"""
Offline batch generation script for evaluation.

Simulates the real-time server's sliding-window inpainting pipeline offline,
generating full-song accompaniment stems from the Slakh2100 test set.
Output matches stream-music-gen's evaluation folder format for COCOLA, Beat F1, FAD.

Outputs go to:
    lightning_logs/streaming_eval_outputs/{run_name}/model_predictions/00000/...

Run name is auto-generated as:
    {config_stem}_r{r}_w{w}[_hot][_{stems}]
or overridden with --run_name.

Usage:
    python generate_eval.py \
        --config configs/for_server/Diff_latent_cond_gen_concat_eval.yaml \
        --r 0.25 --w 0 \
        --num_samples 10 \
        --device cuda:1 \
        --stems bass drums guitar piano
"""

import argparse
import glob
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm


EVAL_SAMPLE_RATE = 16000  # target SR for COCOLA / FAD

# ---------------------------------------------------------------------------
# Utility functions (adapted from server.py)
# ---------------------------------------------------------------------------

STEM_NAMES = ["bass", "drums", "guitar", "piano"]
T_SAMPLES = 264600  # 6 seconds @ 44100 Hz
LATENT_SIZE = 64
SR = 44100


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def instantiate_from_config(config, **kwargs):
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    module_path, class_name = config["_target_"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    config_dict = {k: v for k, v in config.items() if k != "_target_"}
    return cls(**config_dict, **kwargs)


def create_temporal_mask(like, mask_ratio):
    """Creates a temporal mask: True = keep (context), False = generate."""
    _, _, F, T = like.shape
    mask = torch.ones_like(like, dtype=torch.bool)
    t_mask = int(T * mask_ratio)
    t_start = T - t_mask
    mask[:, :, :, t_start:] = False
    return mask


def shift_tensor_data(tensor, percentage):
    """Shifts data left along the last dimension by percentage, zeros the tail."""
    last_dim_size = tensor.size(-1)
    shift_size = int(last_dim_size * percentage)
    if shift_size == 0:
        return
    temp = tensor.clone()
    tensor[..., :-shift_size] = temp[..., shift_size:]
    tensor[..., -shift_size:] = 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path, checkpoint_path, device):
    """Load diffusion model, CAE, sampler, and schedule from config."""
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    cfg = dict2namespace(config)

    # Model
    diffusion_sigma_distribution = instantiate_from_config(cfg.diffusion_sigma_distribution)
    model = instantiate_from_config(cfg.model, diffusion_sigma_distribution=diffusion_sigma_distribution)
    print("Initialized diffusion model.")

    # Checkpoint
    ckpt_path = checkpoint_path or getattr(cfg, "resume_from_checkpoint", None)
    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Checkpoint loaded.")
    else:
        print("WARNING: No checkpoint — running with random weights.")

    model.to(device)
    model.eval()

    # Sampler and schedule
    sampler = instantiate_from_config(cfg.diffusion_sampler)
    schedule = instantiate_from_config(cfg.diffusion_schedule)
    steps = cfg.audio_samples_logger.sampling_steps

    return model, sampler, schedule, steps, config


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_stem(model, sampler, schedule, mixture_audio, stem_idx, r, w,
                  num_steps, device, hot_start=False, gt_audio=None):
    """
    Sliding-window inpainting generation of a single stem for a full song.

    Args:
        model: Audio_LDM_Model (on device, eval mode)
        sampler: diffusion sampler (e.g. ADPM2Sampler)
        schedule: sigma schedule (e.g. KarrasSchedule)
        mixture_audio: (L,) tensor of full context/accompaniment waveform
        stem_idx: 0=bass, 1=drums, 2=guitar, 3=piano
        r: step ratio (e.g. 0.25)
        w: look-ahead depth (-1=offline, 0=sync, 1=look-ahead)
        num_steps: diffusion sampling steps
        device: torch device string
        hot_start: if True, initialize latent buffer with GT stem for first window
        gt_audio: (L,) tensor of ground truth stem (required for hot_start)

    Returns:
        (generated_length,) tensor of generated stem waveform
    """
    step_samples = int(T_SAMPLES * r)
    step_latent = int(LATENT_SIZE * r)
    pr_win_mul = w + 1  # convert w to server's internal parameter
    L = mixture_audio.shape[0]

    # Number of sliding window steps
    n_steps = max(1, (L - T_SAMPLES) // step_samples + 1)

    # Buffers
    latent_buffer = torch.zeros(1, 1, LATENT_SIZE, LATENT_SIZE, device=device)
    mask = create_temporal_mask(latent_buffer, mask_ratio=r).to(device)

    # One-hot stem features
    features = torch.zeros(1, len(STEM_NAMES), device=device)
    features[0, stem_idx] = 1.0

    generated_chunks = []
    chunk_overlaps = []  # overlap_before for each chunk
    decode_deficit = None  # T_SAMPLES - decoded_length (e.g. 920)
    XFADE_MARGIN = 50  # crossfade margin where both sides have real audio

    if hot_start:
        # Hot start: seed latent buffer with GT stem's first window
        gt_window = gt_audio[:T_SAMPLES].clone()
        if gt_window.shape[0] < T_SAMPLES:
            gt_window = torch.nn.functional.pad(gt_window, (0, T_SAMPLES - gt_window.shape[0]))
        latent_buffer = model.CAE.encode(gt_window.unsqueeze(0).to(device)).unsqueeze(1)
        # Prepend GT context (the first (1-r)*T that won't be generated)
        context_samples = T_SAMPLES - step_samples
        generated_chunks.append(gt_audio[:context_samples].cpu())
        chunk_overlaps.append(0)

    for step_i in tqdm(range(n_steps), desc=f"  {STEM_NAMES[stem_idx]}", leave=False):
        # 1. Slice mixture audio for this window
        audio_start = step_i * step_samples
        audio_end = audio_start + T_SAMPLES
        if audio_end > L:
            chunk = torch.zeros(T_SAMPLES)
            valid = L - audio_start
            chunk[:valid] = mixture_audio[audio_start:L]
        else:
            chunk = mixture_audio[audio_start:audio_end]
        tensor = chunk.unsqueeze(0).to(device)  # (1, T_SAMPLES)

        # 2. Encode mixture to latent
        mixture_latent = model.CAE.encode(tensor).unsqueeze(1)  # (1, 1, 64, 64)

        # 3. Zero future mixture based on w (pr_win_mul)
        if pr_win_mul > 0:
            zero_start = max(0, int(LATENT_SIZE * (1 - pr_win_mul * r)))
            mixture_latent[:, :, :, zero_start:] = 0.0

        if not hot_start and step_i == 0:
            # Cold start first step: no valid context, generate full window
            full_mask = torch.zeros_like(latent_buffer, dtype=torch.bool).to(device)
            noise = torch.randn(1, 1, LATENT_SIZE, LATENT_SIZE, device=device)

            samples = model.model.inpaint(
                inpaint=noise,
                inpaint_mask=full_mask,
                noise_labels_s=None,
                sampler=sampler,
                sigma_schedule=schedule,
                num_steps=num_steps,
                class_labels=features,
                augment_labels=None,
                mixture=mixture_latent,
            )

            # Keep full latent as context for next steps
            latent_buffer = samples.clone()

            # Decode and keep full window audio (first chunk, no overlap)
            samples_wav = model.CAE.decode(samples.squeeze(1))
            decode_len = samples_wav.shape[-1]
            if decode_deficit is None:
                decode_deficit = T_SAMPLES - decode_len
            if decode_len < T_SAMPLES:
                samples_wav = torch.nn.functional.pad(
                    samples_wav, (0, T_SAMPLES - decode_len)
                )
            generated_chunks.append(samples_wav[0, :T_SAMPLES].cpu())
            chunk_overlaps.append(0)

            # Shift for next step
            shift_tensor_data(latent_buffer, r)
        else:
            # Normal inpainting step
            noise = torch.randn(1, 1, LATENT_SIZE, LATENT_SIZE, device=device)
            inpaint = torch.where(mask, latent_buffer, noise)

            samples = model.model.inpaint(
                inpaint=inpaint,
                inpaint_mask=mask,
                noise_labels_s=None,
                sampler=sampler,
                sigma_schedule=schedule,
                num_steps=num_steps,
                class_labels=features,
                augment_labels=None,
                mixture=mixture_latent,
            )

            # Update latent buffer with generated region
            gen_start = int(LATENT_SIZE * (1 - r))
            latent_buffer[:, :, :, gen_start:] = samples[:, :, :, gen_start:].clone()

            # Decode
            samples_wav = model.CAE.decode(samples.squeeze(1))
            decode_len = samples_wav.shape[-1]
            if decode_deficit is None:
                decode_deficit = T_SAMPLES - decode_len
            if decode_len < T_SAMPLES:
                samples_wav = torch.nn.functional.pad(
                    samples_wav, (0, T_SAMPLES - decode_len)
                )

            # Extract chunk — extended with overlap when possible (not r=1)
            overlap = decode_deficit + XFADE_MARGIN
            can_extend = (
                decode_deficit > 0
                and step_samples + overlap <= decode_len
                and len(generated_chunks) > 0
            )
            if can_extend:
                audio_chunk = samples_wav[0, T_SAMPLES - step_samples - overlap:T_SAMPLES].cpu()
                chunk_overlaps.append(overlap)
            else:
                audio_chunk = samples_wav[0, T_SAMPLES - step_samples:T_SAMPLES].cpu()
                chunk_overlaps.append(0)
            generated_chunks.append(audio_chunk)

            # Shift latent buffer for next step
            shift_tensor_data(latent_buffer, r)

    # Concatenate chunks with crossfade at overlap boundaries
    # Each extended chunk has `overlap` extra samples at the start that overlap
    # with the previous chunk's tail. The previous tail has `XFADE_MARGIN` real
    # samples followed by `decode_deficit` zeros (from padding). We crossfade
    # over the XFADE_MARGIN region (both sides real) then replace the zeros.
    full_generated = generated_chunks[0]
    for i in range(1, len(generated_chunks)):
        chunk = generated_chunks[i]
        overlap = chunk_overlaps[i]
        if overlap > 0:
            xfade = XFADE_MARGIN
            # Crossfade the first xfade samples of overlap (both sides real audio)
            fade = torch.linspace(0, 1, xfade)
            blended = full_generated[-overlap:-overlap + xfade] * (1 - fade) + chunk[:xfade] * fade
            # After crossfade region: current chunk replaces previous zeros
            full_generated = torch.cat([
                full_generated[:-overlap],
                blended,
                chunk[xfade:],
            ])
        else:
            full_generated = torch.cat([full_generated, chunk])

    return full_generated


# ---------------------------------------------------------------------------
# Batched generation — all stems in one forward pass per window (~4x faster)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_all_stems_batched(model, sampler, schedule, mixture_audio, gt_audios, stems,
                                r, w, num_steps, device, hot_start=False):
    """
    Sliding-window inpainting for all stems simultaneously (batch dim = stems).
    Replaces 4 sequential generate_stem calls with a single batched inpaint call
    per window step, giving ~4x speedup on generation.

    Args:
        mixture_audio: (L,) full mixture waveform
        gt_audios:     dict {stem_name: (L',) tensor} for all stems in `stems`
        stems:         list of stem names to generate (subset of STEM_NAMES)

    Returns:
        dict {stem_name: (L_gen,) generated waveform}
    """
    B = len(stems)
    step_samples = int(T_SAMPLES * r)
    pr_win_mul = w + 1
    L = min(mixture_audio.shape[0], min(gt_audios[s].shape[0] for s in stems))
    n_steps = max(1, (L - T_SAMPLES) // step_samples + 1)

    # Context audio per stem: mixture − gt_stem
    context_audios = [mixture_audio[:L] - gt_audios[s][:L] for s in stems]

    # Batched latent buffers (B, 1, 64, 64)
    latent_buffer = torch.zeros(B, 1, LATENT_SIZE, LATENT_SIZE, device=device)
    mask = create_temporal_mask(latent_buffer[:1], mask_ratio=r).expand(B, -1, -1, -1).to(device)

    # One-hot features (B, 4)
    features = torch.zeros(B, len(STEM_NAMES), device=device)
    for i, s in enumerate(stems):
        features[i, STEM_NAMES.index(s)] = 1.0

    generated_chunks = [[] for _ in range(B)]
    chunk_overlaps  = [[] for _ in range(B)]
    decode_deficits = [None] * B
    XFADE_MARGIN = 50

    if hot_start:
        for i, s in enumerate(stems):
            gt_win = gt_audios[s][:T_SAMPLES].clone()
            if gt_win.shape[0] < T_SAMPLES:
                gt_win = torch.nn.functional.pad(gt_win, (0, T_SAMPLES - gt_win.shape[0]))
            latent_buffer[i:i+1] = model.CAE.encode(gt_win.unsqueeze(0).to(device)).unsqueeze(1)
            generated_chunks[i].append(gt_audios[s][:T_SAMPLES - step_samples].cpu())
            chunk_overlaps[i].append(0)

    for step_i in tqdm(range(n_steps), desc="  [all stems]", leave=False):
        audio_start = step_i * step_samples
        audio_end   = audio_start + T_SAMPLES

        # Build context batch (B, T_SAMPLES) — different context per stem
        ctx_list = []
        for ctx in context_audios:
            if audio_end > L:
                chunk = torch.zeros(T_SAMPLES)
                valid = L - audio_start
                if valid > 0:
                    chunk[:valid] = ctx[audio_start:L]
            else:
                chunk = ctx[audio_start:audio_end]
            ctx_list.append(chunk)
        ctx_batch = torch.stack(ctx_list, dim=0).to(device)          # (B, T_SAMPLES)
        mixture_latent = model.CAE.encode(ctx_batch).unsqueeze(1)    # (B, 1, 64, 64)

        if pr_win_mul > 0:
            zero_start = max(0, int(LATENT_SIZE * (1 - pr_win_mul * r)))
            mixture_latent[:, :, :, zero_start:] = 0.0

        if not hot_start and step_i == 0:
            noise     = torch.randn(B, 1, LATENT_SIZE, LATENT_SIZE, device=device)
            full_mask = torch.zeros(B, 1, LATENT_SIZE, LATENT_SIZE, dtype=torch.bool, device=device)
            samples = model.model.inpaint(
                inpaint=noise, inpaint_mask=full_mask, noise_labels_s=None,
                sampler=sampler, sigma_schedule=schedule, num_steps=num_steps,
                class_labels=features, augment_labels=None, mixture=mixture_latent,
            )
            latent_buffer = samples.clone()
            samples_wav = model.CAE.decode(samples.squeeze(1))   # (B, L_dec)
            dec_len = samples_wav.shape[-1]
            for i in range(B):
                if decode_deficits[i] is None:
                    decode_deficits[i] = T_SAMPLES - dec_len
                wav = samples_wav[i]
                if dec_len < T_SAMPLES:
                    wav = torch.nn.functional.pad(wav, (0, T_SAMPLES - dec_len))
                generated_chunks[i].append(wav[:T_SAMPLES].cpu())
                chunk_overlaps[i].append(0)
            shift_tensor_data(latent_buffer, r)

        else:
            noise   = torch.randn(B, 1, LATENT_SIZE, LATENT_SIZE, device=device)
            inpaint = torch.where(mask, latent_buffer, noise)
            samples = model.model.inpaint(
                inpaint=inpaint, inpaint_mask=mask, noise_labels_s=None,
                sampler=sampler, sigma_schedule=schedule, num_steps=num_steps,
                class_labels=features, augment_labels=None, mixture=mixture_latent,
            )
            gen_start = int(LATENT_SIZE * (1 - r))
            latent_buffer[:, :, :, gen_start:] = samples[:, :, :, gen_start:].clone()
            samples_wav = model.CAE.decode(samples.squeeze(1))   # (B, L_dec)
            dec_len = samples_wav.shape[-1]
            for i in range(B):
                if decode_deficits[i] is None:
                    decode_deficits[i] = T_SAMPLES - dec_len
                wav = samples_wav[i]
                if dec_len < T_SAMPLES:
                    wav = torch.nn.functional.pad(wav, (0, T_SAMPLES - dec_len))
                overlap = decode_deficits[i] + XFADE_MARGIN
                can_extend = (
                    decode_deficits[i] > 0
                    and step_samples + overlap <= dec_len
                    and len(generated_chunks[i]) > 0
                )
                if can_extend:
                    generated_chunks[i].append(wav[T_SAMPLES - step_samples - overlap:T_SAMPLES].cpu())
                    chunk_overlaps[i].append(overlap)
                else:
                    generated_chunks[i].append(wav[T_SAMPLES - step_samples:T_SAMPLES].cpu())
                    chunk_overlaps[i].append(0)
            shift_tensor_data(latent_buffer, r)

    # Assemble per-stem with crossfade (identical logic to generate_stem)
    results = {}
    for i, s in enumerate(stems):
        full = generated_chunks[i][0]
        for j in range(1, len(generated_chunks[i])):
            chunk = generated_chunks[i][j]
            ov    = chunk_overlaps[i][j]
            if ov > 0:
                fade    = torch.linspace(0, 1, XFADE_MARGIN)
                blended = full[-ov:-ov + XFADE_MARGIN] * (1 - fade) + chunk[:XFADE_MARGIN] * fade
                full    = torch.cat([full[:-ov], blended, chunk[XFADE_MARGIN:]])
            else:
                full = torch.cat([full, chunk])
        results[s] = full
    return results


# ---------------------------------------------------------------------------
# Track processing and output saving
# ---------------------------------------------------------------------------

def save_wav(path, audio, sr=SR):
    """Save a 1D tensor as a mono WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, L)
    torchaudio.save(str(path), audio, sr)


def process_track(track_dir, model, sampler, schedule, stems, r, w,
                  num_steps, output_base, sample_counter, device,
                  max_duration=None, hot_start=False):
    """
    Generate all requested stems for one track, save in eval format.

    Returns updated sample_counter.
    """
    track_dir = Path(track_dir)
    track_id = track_dir.name

    # Load mixture
    mixture_path = track_dir / "mixture.wav"
    if not mixture_path.exists():
        print(f"  Skipping {track_id}: no mixture.wav")
        return sample_counter
    mixture_audio, sr = torchaudio.load(str(mixture_path))
    mixture_audio = mixture_audio[0]  # mono (L,)
    if max_duration is not None:
        max_samples = int(max_duration * SR)
        mixture_audio = mixture_audio[:max_samples]

    if mixture_audio.shape[0] < T_SAMPLES:
        print(f"  Skipping {track_id}: too short ({mixture_audio.shape[0]} < {T_SAMPLES})")
        return sample_counter

    step_samples = int(T_SAMPLES * r)

    # Load all available GT stems up front
    gt_audios = {}
    available_stems = []
    for stem_name in stems:
        gt_path = track_dir / f"{stem_name}.wav"
        if not gt_path.exists():
            print(f"  Skipping {track_id}/{stem_name}: no {stem_name}.wav")
            continue
        gt_audio, _ = torchaudio.load(str(gt_path))
        gt_audio = gt_audio[0]
        if max_duration is not None:
            gt_audio = gt_audio[:int(max_duration * SR)]
        gt_audios[stem_name] = gt_audio
        available_stems.append(stem_name)

    if not available_stems:
        return sample_counter

    # If all output dirs for this track already exist, skip generation
    first_sample_dir = output_base / "model_predictions" / f"{sample_counter:05d}"
    if (first_sample_dir / "pred" / "pred.wav").exists():
        print(f"  Skipping {track_id}: already generated (starting at {sample_counter:05d})")
        return sample_counter + len(available_stems)

    # Generate all stems in one batched pass (~4x faster than sequential)
    generated_dict = generate_all_stems_batched(
        model, sampler, schedule, mixture_audio, gt_audios, available_stems,
        r, w, num_steps, device, hot_start=hot_start,
    )

    # Save outputs per stem
    for stem_name in available_stems:
        stem_idx   = STEM_NAMES.index(stem_name)
        generated  = generated_dict[stem_name]
        gen_length = generated.shape[0]

        gt_trimmed    = gt_audios[stem_name][:gen_length]
        mix_trimmed   = mixture_audio[:gen_length]
        accompaniment = mix_trimmed - gt_trimmed
        pred_mix      = accompaniment + generated

        sample_dir = output_base / "model_predictions" / f"{sample_counter:05d}"
        save_wav(sample_dir / "input_audio.wav",        accompaniment)
        save_wav(sample_dir / "ground_truth" / "pred.wav", gt_trimmed)
        save_wav(sample_dir / "ground_truth" / "mix.wav",  mix_trimmed)
        save_wav(sample_dir / "pred" / "pred.wav",         generated)
        save_wav(sample_dir / "pred" / "mix.wav",          pred_mix)

        common_length = min(mixture_audio.shape[0], gt_audios[stem_name].shape[0])
        n_steps_meta  = max(1, (common_length - T_SAMPLES) // step_samples + 1)
        metadata = {
            "track_id": track_id, "stem": stem_name, "stem_idx": stem_idx,
            "r": r, "w": w, "pr_win_mul": w + 1, "hot_start": hot_start,
            "num_steps": n_steps_meta, "sampling_steps": num_steps,
            "duration_s": round(mixture_audio.shape[0] / SR, 2),
            "generated_duration_s": round(gen_length / SR, 2),
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"  Saved {sample_dir.name}: {track_id}/{stem_name} "
              f"({gen_length / SR:.1f}s, {n_steps_meta} steps)")
        sample_counter += 1

    return sample_counter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STREAMING_EVAL_BASE = Path("lightning_logs/streaming_eval_outputs")


def make_run_name(config_path, r, w, hot_start, stems):
    """Auto-generate a run name from config + hyperparams."""
    # e.g. configs/for_server/Diff_latent_cond_gen_concat_eval.yaml
    #   -> Diff_latent_cond_gen_concat
    stem = Path(config_path).stem
    for suffix in ("_eval", "_train"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    name = f"{stem}_r{r}_w{w}"
    if hot_start:
        name += "_hot"
    if sorted(stems) != sorted(STEM_NAMES):
        name += "_" + "+".join(stems)
    return name


def parse_args():
    parser = argparse.ArgumentParser(description="Offline generation for evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: from config)")
    parser.add_argument("--r", type=float, default=0.25,
                        help="Step ratio (default: 0.25)")
    parser.add_argument("--w", type=int, default=0,
                        help="Look-ahead depth: -1=offline, 0=sync, 1=look-ahead")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max number of test tracks (default: all)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override auto-generated run name")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")
    parser.add_argument("--stems", nargs="+", default=STEM_NAMES,
                        choices=STEM_NAMES,
                        help="Which stems to generate (default: all 4)")
    parser.add_argument("--test_data_dir", type=str,
                        default="dataset/slakh2100_44100/test",
                        help="Path to test data directory")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Limit song duration in seconds (for debugging)")
    parser.add_argument("--hot_start", action="store_true",
                        help="Initialize first window with GT stem latent")

    # --- evaluation flags (mirrors stream-music-gen gen_and_evaluate.py) ---
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, run evaluation only on existing run folder")
    parser.add_argument("--skip_resampling", action="store_true",
                        help="Skip resampling to 16 kHz (assumes already done)")
    parser.add_argument("--skip_beat_alignment", action="store_true",
                        help="Skip beat alignment score calculation")
    parser.add_argument("--skip_cocola", action="store_true",
                        help="Skip COCOLA score calculation")
    parser.add_argument("--skip_fad", action="store_true",
                        help="Skip FAD calculation")
    parser.add_argument("--allow_degenerate", action="store_true",
                        help="Run even degenerate configs (>50%% mixture zeroed, or w=-1 with r<0.125)")
    parser.add_argument("--sub_fad", action="store_true",
                        help="Evaluate FAD on mixes instead of stems")
    parser.add_argument("--generation_only", action="store_true",
                        help="Only generate audio, skip evaluation")
    parser.add_argument("--keep_audio", action="store_true",
                        help="Keep model_predictions/ after evaluation (default: delete to save disk)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to W&B")
    parser.add_argument("--wandb_project", type=str, default="stream-music-gen",
                        help="W&B project name (default: stream-music-gen)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (default: logged-in user)")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Resume an existing W&B run by ID")
    return parser.parse_args()


def flatten_results(results, prefix=""):
    """Flatten nested results dict into wandb-loggable key/value pairs."""
    flat = {}
    for k, v in results.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_results(v, prefix=key))
        else:
            flat[key] = v
    return flat


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # W&B init — before run_name so sweep agent can override r / w
    # -----------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        import wandb as _wandb

        T_s = T_SAMPLES / SR
        STREAMING_EVAL_BASE.mkdir(parents=True, exist_ok=True)
        wandb_run = _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            id=args.wandb_run_id or None,
            resume="must" if args.wandb_run_id else None,
            dir=str(STREAMING_EVAL_BASE),
            config={
                # core hyperparams
                "r":            args.r,
                "w":            args.w,
                "hot_start":    args.hot_start,
                # derived timing (comparable to streaming delay times)
                "T_s":          T_s,
                "step_s":       round(T_s * args.r, 4),
                "context_s":    round(T_s * (1 - args.r), 4),
                "lookahead_s":  round(T_s * args.r * max(0, args.w), 4),
                "net_lookahead_s": round(T_s * args.r * args.w, 4),
                # run setup
                "config":       args.config,
                "checkpoint":   args.checkpoint,
                "stems":        args.stems,
                "num_samples":  args.num_samples,
                "max_duration": args.max_duration,
                "test_data_dir": args.test_data_dir,
            },
        )
        # Allow sweep agent to override r / w
        args.r = _wandb.config.get("r", args.r)
        args.w = _wandb.config.get("w", args.w)

    # ------------------------------------------------------------------
    # Skip degenerate sweep configurations (override with --allow_degenerate)
    # ------------------------------------------------------------------
    pr_win_mul = args.w + 1
    skip_reason = None
    if not args.allow_degenerate and pr_win_mul > 0 and pr_win_mul * args.r > 0.5:
        skip_reason = (f"w={args.w}, r={args.r}: {pr_win_mul}*{args.r}={pr_win_mul * args.r:.4f} > 0.5 "
                       f"(more than half of mixture context zeroed)")
    elif not args.allow_degenerate and args.w == -1 and args.r < 0.125:
        skip_reason = (f"w=-1, r={args.r}: offline mode with r<0.125 not informative "
                       f"(diminishing returns, high compute cost)")
    if skip_reason:
        print(f"Skipping degenerate config: {skip_reason}")
        if wandb_run is not None:
            wandb_run.notes = f"skipped: {skip_reason}"
            wandb_run.finish(exit_code=0)
        sys.exit(0)

    run_name = args.run_name or make_run_name(args.config, args.r, args.w,
                                              args.hot_start, args.stems)
    if wandb_run is not None:
        wandb_run.name = run_name

    output_base = STREAMING_EVAL_BASE / run_name

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    if not args.skip_generation:
        print(f"Config: {args.config}")
        print(f"r={args.r}, w={args.w} (pr_win_mul={args.w + 1})")
        print(f"Stems: {args.stems}")
        print(f"Device: {args.device}")
        print(f"Run:    {run_name}")
        print(f"Output: {output_base}")

        model, sampler, schedule, sampling_steps, _ = load_model(
            args.config, args.checkpoint, args.device
        )

        test_dir = Path(args.test_data_dir)
        track_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        if args.num_samples is not None:
            track_dirs = track_dirs[:args.num_samples]

        print(f"Test tracks: {len(track_dirs)}")
        print(f"Sampling steps: {sampling_steps}")

        output_base.mkdir(parents=True, exist_ok=True)

        predictions_dir = output_base / "model_predictions"
        sample_counter = len([d for d in predictions_dir.iterdir() if d.is_dir()]) if predictions_dir.exists() else 0
        if sample_counter > 0:
            print(f"Resuming: found {sample_counter} existing samples, skipping already-done tracks.")

        for track_dir in tqdm(track_dirs, desc="Tracks"):
            sample_counter = process_track(
                track_dir, model, sampler, schedule,
                args.stems, args.r, args.w, sampling_steps,
                output_base, sample_counter, args.device,
                max_duration=args.max_duration,
                hot_start=args.hot_start
            )

        print(f"\nDone! Generated {sample_counter} samples in {output_base}")

    if args.generation_only:
        return

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    from main.eval import eval_utils
    from main.eval.beat_alignment_eval import beat_alignment_score
    import main.eval.cocola_eval as cocola_eval
    import main.eval.fad_eval as fad_eval

    results = {}
    root_folder = str(output_base / "model_predictions")

    if not args.skip_resampling:
        print("\nResampling audio to 16 kHz...")
        files_to_resample = glob.glob(
            os.path.join(root_folder, "**", "*.wav"), recursive=True
        )
        # exclude already-resampled files
        files_to_resample = [f for f in files_to_resample
                             if f"_{EVAL_SAMPLE_RATE}.wav" not in f]
        eval_utils.load_resample_save(files_to_resample, SR, EVAL_SAMPLE_RATE, batch_size=1)

    if not args.skip_beat_alignment:
        print("\nComputing beat alignment...")
        all_gt_scores, all_pred_scores = beat_alignment_score(
            root_folder,
            context_path="input_audio.wav",
            gt_path="ground_truth/pred.wav",
            pred_path="pred/pred.wav",
        )
        f_key = "madmom_fmeasure"
        if f_key in all_gt_scores:
            gt_f = all_gt_scores[f_key]
            pred_f = all_pred_scores[f_key]
            results["beat_alignment"] = {
                "gt_f_measure":   {"mean": float(np.mean(gt_f)),   "std": float(np.std(gt_f))},
                "pred_f_measure": {"mean": float(np.mean(pred_f)), "std": float(np.std(pred_f))},
            }
            print(f"GT F-Measure:   mean={np.mean(gt_f):.4f}  std={np.std(gt_f):.4f}")
            print(f"Pred F-Measure: mean={np.mean(pred_f):.4f}  std={np.std(pred_f):.4f}")

    if not args.skip_cocola:
        print("\nComputing COCOLA...")
        embedding_modes = ["both", "harmonic", "percussive"]
        gt_scores, pred_scores = cocola_eval.cocola_score(
            root_folder,
            context_path=f"input_audio_{EVAL_SAMPLE_RATE}.wav",
            gt_path=f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav",
            pred_path=f"pred/pred_{EVAL_SAMPLE_RATE}.wav",
            embedding_modes=embedding_modes,
        )
        results["cocola"] = {}
        for mode in embedding_modes:
            gt_m, pred_m = gt_scores[mode], pred_scores[mode]
            results["cocola"][mode] = {
                "gt_scores":   {"mean": float(np.mean(gt_m)), "std": float(np.std(gt_m))},
                "pred_scores": {"mean": float(np.mean(pred_m)), "std": float(np.std(pred_m))},
            }
            print(f"COCOLA [{mode}]: GT={np.mean(gt_m):.4f}  Pred={np.mean(pred_m):.4f}")

    if not args.skip_fad:
        print("\nComputing FAD...")
        if args.sub_fad:
            gt_path   = f"ground_truth/mix_{EVAL_SAMPLE_RATE}.wav"
            gen_path  = f"pred/mix_{EVAL_SAMPLE_RATE}.wav"
        else:
            gt_path   = f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav"
            gen_path  = f"pred/pred_{EVAL_SAMPLE_RATE}.wav"
        fad_score = fad_eval.calculate_fad(root_folder, gt_path=gt_path,
                                           gen_path=gen_path, metric="vggish")
        results["fad"] = {"score": float(fad_score)}
        print(f"FAD: {fad_score:.4f}")

    # Save results inside the run folder
    results_file = output_base / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    if wandb_run is not None:
        import wandb as _wandb
        _wandb.log(flatten_results(results))
        wandb_run.finish()

    # Clean up generated audio unless --keep_audio
    if not args.keep_audio:
        predictions_dir = output_base / "model_predictions"
        if predictions_dir.exists():
            shutil.rmtree(str(predictions_dir))
            print(f"Cleaned up {predictions_dir} (use --keep_audio to retain audio files)")


if __name__ == "__main__":
    main()
