"""
Offline batch generation script for evaluation.

Simulates the real-time server's sliding-window inpainting pipeline offline,
generating full-song accompaniment stems from the Slakh2100 test set.
Output matches stream-music-gen's evaluation folder format for COCOLA, Beat F1, FAD.

Usage:
    python generate_eval.py \
        --config configs/for_server/Diff_latent_cond_gen_concat_eval.yaml \
        --r 0.25 --w 0 \
        --num_samples 10 \
        --output_dir lightning_logs/eval_outputs/diffusion_r0.25_w0 \
        --device cuda:1 \
        --stems bass drums guitar piano
"""

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import torchaudio
import yaml
from tqdm import tqdm


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

    if hot_start:
        # Hot start: seed latent buffer with GT stem's first window
        gt_window = gt_audio[:T_SAMPLES].clone()
        if gt_window.shape[0] < T_SAMPLES:
            gt_window = torch.nn.functional.pad(gt_window, (0, T_SAMPLES - gt_window.shape[0]))
        latent_buffer = model.CAE.encode(gt_window.unsqueeze(0).to(device)).unsqueeze(1)
        # Prepend GT context (the first (1-r)*T that won't be generated)
        context_samples = T_SAMPLES - step_samples
        generated_chunks.append(gt_audio[:context_samples].cpu())

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

            # Decode and keep full window audio
            samples_wav = model.CAE.decode(samples.squeeze(1))
            if samples_wav.shape[-1] < T_SAMPLES:
                samples_wav = torch.nn.functional.pad(
                    samples_wav, (0, T_SAMPLES - samples_wav.shape[-1])
                )
            generated_chunks.append(samples_wav[0, :T_SAMPLES].cpu())

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

            # Decode and extract the newly generated chunk
            samples_wav = model.CAE.decode(samples.squeeze(1))
            if samples_wav.shape[-1] < T_SAMPLES:
                samples_wav = torch.nn.functional.pad(
                    samples_wav, (0, T_SAMPLES - samples_wav.shape[-1])
                )
            audio_chunk = samples_wav[0, T_SAMPLES - step_samples:T_SAMPLES].cpu()
            generated_chunks.append(audio_chunk)

            # Shift latent buffer for next step
            shift_tensor_data(latent_buffer, r)

    # Concatenate all chunks
    full_generated = torch.cat(generated_chunks, dim=0)
    return full_generated


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

    for stem_name in stems:
        stem_idx = STEM_NAMES.index(stem_name)

        # Load ground truth stem
        gt_path = track_dir / f"{stem_name}.wav"
        if not gt_path.exists():
            print(f"  Skipping {track_id}/{stem_name}: no {stem_name}.wav")
            continue

        gt_audio, _ = torchaudio.load(str(gt_path))
        gt_audio = gt_audio[0]  # mono (L,)
        if max_duration is not None:
            gt_audio = gt_audio[:int(max_duration * SR)]

        # Use minimum length of mixture and gt stem
        common_length = min(mixture_audio.shape[0], gt_audio.shape[0])

        # Context = accompaniment (mixture minus target stem)
        context_audio = mixture_audio[:common_length] - gt_audio[:common_length]

        # Generate conditioned on context (what the performer plays)
        generated = generate_stem(
            model, sampler, schedule,
            context_audio, stem_idx, r, w, num_steps, device,
            hot_start=hot_start,
            gt_audio=gt_audio[:common_length] if hot_start else None,
        )

        # Use actual generated length for trimming
        gen_length = generated.shape[0]
        gt_trimmed = gt_audio[:gen_length]
        mix_trimmed = mixture_audio[:gen_length]
        accompaniment = mix_trimmed - gt_trimmed  # input = mixture minus target stem

        # Construct pred mix: accompaniment + generated stem
        pred_mix = accompaniment + generated

        # Output folder
        sample_dir = output_base / "model_predictions" / f"{sample_counter:05d}"

        # Save files
        save_wav(sample_dir / "input_audio.wav", accompaniment)
        save_wav(sample_dir / "ground_truth" / "pred.wav", gt_trimmed)
        save_wav(sample_dir / "ground_truth" / "mix.wav", mix_trimmed)
        save_wav(sample_dir / "pred" / "pred.wav", generated)
        save_wav(sample_dir / "pred" / "mix.wav", pred_mix)

        # Metadata
        n_steps = max(1, (common_length - T_SAMPLES) // step_samples + 1)
        metadata = {
            "track_id": track_id,
            "stem": stem_name,
            "stem_idx": stem_idx,
            "r": r,
            "w": w,
            "pr_win_mul": w + 1,
            "hot_start": hot_start,
            "num_steps": n_steps,
            "sampling_steps": num_steps,
            "duration_s": round(mixture_audio.shape[0] / SR, 2),
            "generated_duration_s": round(gen_length / SR, 2),
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"  Saved {sample_dir.name}: {track_id}/{stem_name} "
              f"({gen_length / SR:.1f}s, {n_steps} steps)")

        sample_counter += 1

    return sample_counter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated audio")
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Config: {args.config}")
    print(f"r={args.r}, w={args.w} (pr_win_mul={args.w + 1})")
    print(f"Stems: {args.stems}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    # Load model
    model, sampler, schedule, sampling_steps, config = load_model(
        args.config, args.checkpoint, args.device
    )

    # List test tracks
    test_dir = Path(args.test_data_dir)
    track_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if args.num_samples is not None:
        track_dirs = track_dirs[:args.num_samples]

    print(f"Test tracks: {len(track_dirs)}")
    print(f"Sampling steps: {sampling_steps}")

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    sample_counter = 0
    for track_dir in tqdm(track_dirs, desc="Tracks"):
        sample_counter = process_track(
            track_dir, model, sampler, schedule,
            args.stems, args.r, args.w, sampling_steps,
            output_base, sample_counter, args.device,
            max_duration=args.max_duration,
            hot_start=args.hot_start
        )

    print(f"\nDone! Generated {sample_counter} samples in {output_base}")


if __name__ == "__main__":
    main()
