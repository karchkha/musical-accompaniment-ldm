"""
multi_track OSC server — diffusion-based musical accompaniment generation.

Receives audio mixture chunks from Max/MSP via OSC, runs inpainting with a
latent diffusion model, and streams predicted stem waveforms back in enumerated
chunks for reassembly on the Max side.

If launching returns 'Address already in use':
  Run 'sudo lsof -i:7000' (or whichever port) to find the occupying PID,
  then 'kill <PID>' to free it.

Author: Tornike Karchkhadze  tkarchkhadze@ucsd.edu
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import platform
import socket
import struct
import warnings
import time
import sys
import importlib
from threading import Thread, Lock, Timer
from queue import Queue

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torchaudio
import yaml

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend — safe for server use
import matplotlib.pyplot as plt

from pythonosc import dispatcher, osc_server, udp_client
from pythonosc import osc_bundle_builder, osc_message_builder
from pytorch_lightning import seed_everything
from music2latent import EncoderDecoder

# ── Project ───────────────────────────────────────────────────────────────────
sys.path.append("src")
from main.module_base import Audio_LDM_Model  # required for checkpoint deserialization

# ── Suppress noisy upstream warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning,  message=".*flash attention.*")

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Global state
# =============================================================================

# Input mixture buffer (1 track × 6 seconds @ 44100 Hz)
context_audio = torch.full((1, 264600), 0.0)

# Latent representation and inpainting mask (updated each predict cycle)
generated_latent = mask = torch.full((1, 1, 64, 64), 0.0)

# Context latent — CAE encoding of context_audio, recomputed each predict cycle
context_latent = torch.full((1, 1, 64, 64), 0.0)

# Last generated audio — kept for shift_tensor_data and debug export
generated_audio = torch.full((1, 264600), 0.0)

# Model and sampler handles (populated by load_network)
latent_diffusion   = None
diffusion_sampler  = None
diffusion_schedule = None
CAE                = EncoderDecoder(device=device)  # standalone music2latent codec

# Runtime parameters (can be updated via OSC at any time)
steps        = 10
config       = {}
package_size      = 5120   # floats per UDP chunk — tune from Max with /update_package_size
r        = 0.25   # fraction of the window to inpaint
w        = 1.0    # prediction window multiplier
headroom_ratio    = 0.02   # crossfade overlap as fraction of sampling rate — synced from Max fade parameter

stem_names         = ["bass", "drums", "guitar", "piano"]
stems_to_inpaint   = []
stemidx_to_inpaint = []

verbose     = 0     # 0 = silent, 1 = print timing events (toggled via /verbose from Max)
_loading    = False # True while load_network is running; all other handlers ignore messages

filename = "configs/for_server/Diff_latent_cond_gen_concat_eval.yaml"

# Batch placeholder passed to model helpers
batch = [
    torch.full((1, 1, 264600), 0.0),
    torch.zeros(1, 4),
    torch.zeros((1, 4, 264600)),
]

# Per-stem output waveform cache — keys populated from config stems after load_network
waveforms = {s: None for s in stem_names}


# =============================================================================
# Event timer — per-stage latency logging
# =============================================================================

class EventTimer:
    """Lightweight checkpoint timer for logging per-stage latencies."""

    def __init__(self):
        self.checkpoints = []

    def record_event(self, event_name="Event"):
        current_time = time.time()
        ms = int((current_time % 1) * 1000)
        lt = time.localtime(current_time)
        clock = f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"
        if self.checkpoints:
            elapsed = current_time - self.checkpoints[-1][1]
            print(f"  {event_name}: +{elapsed*1000:.1f}ms  {clock}")
        else:
            print(f"  {event_name} (start)  {clock}")
        self.checkpoints.append((event_name, current_time))

    def get_intervals(self):
        return [(self.checkpoints[i][0],
                 self.checkpoints[i][1] - self.checkpoints[i - 1][1])
                for i in range(1, len(self.checkpoints))]


timer = EventTimer()


def _hms() -> str:
    """Return current wall-clock time as HH:MM:SS.mmm string."""
    t  = time.time()
    ms = int((t % 1) * 1000)
    lt = time.localtime(t)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"


# =============================================================================
# Fast OSC datagram builder
# =============================================================================

def _make_osc_dgram(address: str, batch_id: int, chunk_idx: int, total_chunks: int,
                    floats_chunk: np.ndarray) -> bytes:
    """Build a raw OSC datagram: /address [batch_id:i] [chunk_idx:i] [total_chunks:i] [floats:f...].

    ~100x faster than OscMessageBuilder: uses numpy byte packing instead of
    a Python loop with per-element add_arg() calls.
    """
    import struct as _struct
    addr = address.encode() + b'\x00'
    addr += b'\x00' * ((4 - len(addr) % 4) % 4)
    n    = len(floats_chunk)
    tag  = (',iii' + 'f' * n).encode() + b'\x00'
    tag += b'\x00' * ((4 - len(tag) % 4) % 4)
    header = _struct.pack('>iii', batch_id, chunk_idx, total_chunks)
    return addr + tag + header + floats_chunk.astype('>f4').tobytes()


# =============================================================================
# Config / model utilities
# =============================================================================

def dict2namespace(config):
    """Recursively convert a config dict to an argparse.Namespace."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key,
                dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def instantiate_from_config(config, **kwargs):
    """Instantiate a class specified by config['_target_'] with remaining keys as kwargs."""
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    module_path, class_name = config['_target_'].rsplit('.', 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    config_dict = {k: v for k, v in config.items() if k != '_target_'}
    return cls(**config_dict, **kwargs)


def create_temporal_mask(like, mask_ratio):
    """Create a boolean mask that covers the last `mask_ratio` fraction of the time axis.

    Shape: same as `like` (B, C, F, T).  False = masked region to inpaint.
    """
    _, _, F, T = like.shape
    mask = torch.ones_like(like, dtype=torch.bool)
    t_mask  = int(T * mask_ratio)
    t_start = T - t_mask
    mask[:, :, :, t_start:] = False
    return mask


# =============================================================================
# Model loading
# =============================================================================

def load_network(unused_addr):
    """OSC handler for /load_model — loads checkpoint and prepares for inference."""
    global latent_diffusion, stemidx_to_inpaint, steps, context_audio, generated_audio, MSAProc
    global generated_latent, mask, config, package_size, filename
    global r, diffusion_sampler, diffusion_schedule, w
    global _loading
    _loading = True

    config = yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
    cfg    = dict2namespace(config)

    # Instantiate diffusion model
    diffusion_sigma_distribution = instantiate_from_config(cfg.diffusion_sigma_distribution)
    latent_diffusion = instantiate_from_config(
        cfg.model, diffusion_sigma_distribution=diffusion_sigma_distribution)
    print("\nDiffusion model instantiated.")

    # Load checkpoint weights
    checkpoint_path = cfg.resume_from_checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint path — running with random initialization.")

    latent_diffusion.to(device)
    latent_diffusion.eval()

    # Instantiate sampler and schedule
    diffusion_sampler  = instantiate_from_config(cfg.diffusion_sampler)
    diffusion_schedule = instantiate_from_config(cfg.diffusion_schedule)
    steps              = cfg.audio_samples_logger.sampling_steps

    # Pre-compute inpainting mask and initial latent encoding
    mask   = create_temporal_mask(mask, mask_ratio=r).to(device)
    generated_latent = CAE.encode(generated_audio).unsqueeze(1)

    _loading = False
    print("Model ready!\n")


# =============================================================================
# Prediction
# =============================================================================

def predict(*args):
    """Run one inpainting prediction cycle and stream results back to Max via OSC."""
    global latent_diffusion, context_audio, waveforms, stems_to_inpaint, stemidx_to_inpaint
    global steps, batch, package_size, r, config
    global diffusion_sampler, diffusion_schedule, generated_latent, context_latent
    global _first_chunk_time, _last_chunk_time, _chunk_count

    t_predict_received = time.time()

    # Snapshot batch_id now — _current_batch_id may be overwritten by a new incoming
    # batch while inference is running (~1-2s), causing response chunks to be tagged
    # with the wrong id and written into the wrong buffer window.
    response_batch_id = _current_batch_id

    # Guard against concurrent predictions
    if not _predict_sem.acquire(blocking=False):
        print(f"[PREDICT] skipped — already running  batch={response_batch_id}  {_hms()}")
        client.send_message("/batch_dropped", int(response_batch_id))
        return

    if verbose:
        gap_since_first = (t_predict_received - _first_chunk_time) if _first_chunk_time else -1
        gap_since_last  = (t_predict_received - _last_chunk_time)  if _last_chunk_time  else -1
        print("----------------------------------------")
        print(f"[PREDICT] batch={response_batch_id}  triggered  {_chunk_count} chunks  "
              f"first {gap_since_first*1000:.1f}ms ago  last {gap_since_last*1000:.1f}ms ago  {_hms()}")

    _sem_released = False
    try:
        # Drain the incoming data queue before touching the tensor
        message_queue.join()
        if verbose:
            print(f"  queue drained  +{(time.time() - t_predict_received)*1000:.1f}ms  {_hms()}")

        timer.checkpoints.clear()
        if verbose:
            timer.record_event("Predict start")

        with torch.no_grad():
            noise = torch.randn(
                (1,
                 config['audio_samples_logger']['channels'],
                 latent_diffusion.model.diffusion.net.img_resolution,
                 latent_diffusion.model.diffusion.net.img_resolution),
                device=latent_diffusion.device
            )

            # One-hot feature vector: 1 = preserved stem, 0 = generated stem
            current_features = torch.zeros(
                1, len(config['audio_samples_logger']['stems']),
                device=latent_diffusion.device)
            for idx in stemidx_to_inpaint:
                current_features[:, idx] = 1

            # Encode mixture and zero out the prediction window
            context_latent = CAE.encode(context_audio).unsqueeze(1)
            if verbose: timer.record_event("Mixture latent encoded")

            start_idx = int(context_latent.size(-1) * (1 - (w + 1) * r))
            context_latent[:, :, :, start_idx:] = 0.0

            # Inject noise into the masked region of the stored latent
            inpaint = generated_latent.clone()
            inpaint = torch.where(mask, inpaint, noise.to(inpaint.dtype))
            if verbose: timer.record_event("Entering sampler")

            # Diffusion inpainting
            samples = latent_diffusion.model.inpaint(
                inpaint=inpaint,
                inpaint_mask=mask,
                noise_labels_s=None,
                sampler=diffusion_sampler,
                sigma_schedule=diffusion_schedule,
                num_steps=steps,
                class_labels=current_features,
                augment_labels=None,
                mixture=context_latent,
            )
            if verbose: timer.record_event("Sampling done")

            # Update generated latent with the freshly inpainted region
            start_idx = int(samples.size(-1) * (1 - r))
            generated_latent[:, :, :, start_idx:] = samples[:, :, :, start_idx:].clone()

        # Shift context_audio and generated_latent so the next batch's incoming data lands at correct positions,
        # then release the semaphore — decode and send don't touch shared state.
        shift_size = int(context_audio.size(-1) * r)
        shift_tensor_data(context_audio, r)
        shift_tensor_data(generated_latent, r)
        if verbose:
            print(f"  Buffers shifted left by {shift_size} samples.  {_hms()}")
            timer.record_event("Buffers shifted")
        _predict_sem.release()
        _sem_released = True

        # Decode — uses only `samples` (local tensor), context_audio/generated_latent no longer needed
        total_length     = config['audio_samples_logger']['length']
        headroom_samples = int(headroom_ratio * config['audio_samples_logger']['sampling_rate'])
        n_needed         = int(total_length * r) + headroom_samples
        n_cols           = int(64 * r) + 1   # e.g. 0.25 → 17, 0.5 → 33
        samples_per_col  = total_length / 64           # 4134.375 — architectural constant
        expected_len     = int(n_cols * samples_per_col)

        with torch.no_grad():
            samples_wav = CAE.decode(
                samples[:, :, :, -n_cols:].squeeze(1)).unsqueeze(1)
            if verbose: timer.record_event("Decoded to waveform")

        # Crop right-edge boundary artifact to align output with original audio grid,
        # then extract exactly the send window from the right
        samples_wav = samples_wav[:, :, :expected_len]
        samples_wav = samples_wav[:, :, -n_needed:]

        samples_wav    = samples_wav.cpu().numpy()
        fade_in_window = np.linspace(0, 1, headroom_samples)

        # Send each predicted stem back to Max as enumerated OSC chunks
        dest = (client._address, client._port)

        for i, stem_name in enumerate(stem_names):
            if i not in stemidx_to_inpaint:
                continue

            if verbose: print(f"  [SEND]    {stem_name}  batch={response_batch_id}  {_hms()}")
            flatten_prediction = samples_wav.flatten().astype(np.float32)

            # Fade-in at the start of the send window to reduce boundary clicks
            flatten_prediction[:headroom_samples] *= fade_in_window

            # Keep generated_audio in sync for debug export
            buf_start = total_length - n_needed
            generated_audio[:, buf_start:buf_start + n_needed] = torch.tensor(flatten_prediction)

            # Stream chunks — each packet carries chunk_idx + total_chunks for reassembly
            chunk_starts = list(range(0, n_needed, package_size))
            total_chunks = len(chunk_starts)
            for chunk_idx, j in enumerate(chunk_starts):
                chunk = flatten_prediction[j : j + min(package_size, n_needed - j)]
                client._sock.sendto(
                    _make_osc_dgram("/" + stem_name, response_batch_id, chunk_idx, total_chunks, chunk),
                    dest)
                # time.sleep(0.0001)  # brief pacing for remote server stability

        if verbose: timer.record_event("Send complete")
        client.send_message("/server_predicted", True)

        shift_tensor_data(generated_audio, r)

    except Exception as e:
        print(f"[PREDICT] ERROR: {e}")
    finally:
        if not _sem_released:
            _predict_sem.release()


# =============================================================================
# Incoming data queue — buffers OSC chunks into the mixture tensor
# =============================================================================

message_queue = Queue()


def process_message_queue():
    """Worker thread: dequeues incoming audio chunks and writes them into `context_audio`."""
    global context_audio, config, r, w
    while True:
        track_id, start_index, values = message_queue.get()

        depth    = context_audio.size(-1)
        # Target write window: [100 - (w+1)*pct, 100 - pct] of the buffer
        start_idx  = int(depth * (1 - (w + 2) * r))
        end_idx    = int(depth * (1 - r))
        range_start = start_idx + start_index
        range_end   = range_start + len(values)

        track_id = 0  # single-track mode — mixture is always track 0
        context_audio[track_id, range_start:range_end] = torch.tensor(values)

        message_queue.task_done()


# Small pool — one thread per expected concurrent chunk stream is plenty
_NUM_QUEUE_WORKERS = 4
for _ in range(_NUM_QUEUE_WORKERS):
    Thread(target=process_message_queue, daemon=True).start()


# =============================================================================
# Watchdog + auto-trigger state
# =============================================================================

_first_chunk_time     = None
_last_chunk_time      = None
_chunk_count          = 0
_auto_lock            = Lock()
_auto_chunks_received = 0
_auto_chunks_expected = 0
_current_batch_id     = -1
_batch_triggered      = False           # True once predict has fired for current batch
_predict_sem          = Lock()          # prevents concurrent predictions
_watchdog_timer       = None
_WATCHDOG_SAFETY_FACTOR = 5             # fire after 5× the observed avg inter-chunk gap


def _watchdog_fire(batch_id):
    """Called when chunks stop arriving before the expected count is reached."""
    global _auto_chunks_received, _batch_triggered
    with _auto_lock:
        if _current_batch_id != batch_id or _auto_chunks_received == 0 or _batch_triggered:
            return  # batch already completed, superseded, or already triggered
        missing = _auto_chunks_expected - _auto_chunks_received
        print(f"WARNING: batch {batch_id} missing {missing}/{_auto_chunks_expected} "
              f"chunks — triggering predict anyway")
        _batch_triggered      = True
        _auto_chunks_received = 0
    Thread(target=predict, daemon=True).start()


def buffer_handler(unused_addr, track_id, batch_id, start_index,
                   total_expected_chunks, *values):
    """OSC handler for incoming audio chunks (/bass, /drums, /guitar, /piano).

    Queues data for tensor population and triggers predict() when all chunks
    for a batch have arrived. An adaptive watchdog fires predict() early if
    chunks stop arriving (e.g. due to UDP packet loss over internet).
    """
    global _first_chunk_time, _last_chunk_time, _chunk_count
    global _auto_chunks_received, _auto_chunks_expected, _current_batch_id, _batch_triggered, _watchdog_timer

    t                    = time.time()
    track_id             = int(track_id[0])
    start_index          = int(start_index)
    batch_id             = int(batch_id)
    total_expected_chunks = int(total_expected_chunks)

    message_queue.put((track_id, start_index, values))

    with _auto_lock:
        if batch_id != _current_batch_id:
            _current_batch_id     = batch_id
            _batch_triggered      = False
            _auto_chunks_received = 0
            _first_chunk_time     = t
            _chunk_count          = 0
            if verbose:
                print(f"[RX]      first chunk  batch={batch_id}  "
                      f"expecting {total_expected_chunks} chunks  {_hms()}")
        elif _batch_triggered:
            return  # late chunk for already-triggered batch — discard

        _auto_chunks_expected  = total_expected_chunks
        _auto_chunks_received += 1
        _chunk_count          += 1
        _last_chunk_time       = t
        should_predict         = (_auto_chunks_received >= _auto_chunks_expected)

        if should_predict:
            _batch_triggered      = True
            _auto_chunks_received = 0
            if _watchdog_timer:
                _watchdog_timer.cancel()
                _watchdog_timer = None
        else:
            # Adaptive watchdog: timeout = avg inter-chunk interval × safety factor
            if _watchdog_timer:
                _watchdog_timer.cancel()
            if _auto_chunks_received > 1:
                avg_interval = ((_last_chunk_time - _first_chunk_time)
                                / (_auto_chunks_received - 1))
            else:
                avg_interval = 0.05  # conservative fallback before 2 samples
            timeout = avg_interval * _WATCHDOG_SAFETY_FACTOR
            _watchdog_timer = Timer(timeout, _watchdog_fire, args=(batch_id,))
            _watchdog_timer.start()

    if should_predict:
        Thread(target=predict, daemon=True).start()


# =============================================================================
# Tensor utilities
# =============================================================================

def shift_tensor_data(tensor: torch.Tensor, r: float):
    """Shift the last dimension left by `r`, zero-filling the tail.

    Used to slide the rolling mixture buffer and latent forward after each
    prediction cycle so the next window is positioned correctly.
    """
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"r must be 0–1, got {r}")

    shift_size = int(tensor.size(-1) * r)
    if shift_size == 0:
        return

    temp = tensor.clone()
    tensor[..., :-shift_size] = temp[..., shift_size:]
    tensor[..., -shift_size:] = 0.0


# =============================================================================
# OSC handlers — control messages
# =============================================================================

def handle_predict_instruments(address, *args):
    """OSC /predict_instruments — one-hot vector, length = number of stems in config."""
    global stemidx_to_inpaint, stems_to_inpaint

    if len(args) != len(stem_names):
        print(f"predict_instruments: got {len(args)} flags but config has {len(stem_names)} stems")
    stems_to_inpaint   = [stem_names[i] for i in range(min(len(args), len(stem_names))) if args[i] == 1]
    stemidx_to_inpaint = [i for i, s in enumerate(stem_names) if s in stems_to_inpaint]
    print(f"Stems to generate: {stems_to_inpaint}")


def reset_tensor(unused_addr, *args):
    """OSC /reset — zero out mixture buffer, stored latent, and generated audio (full restart)."""
    global context_audio, context_latent, generated_latent, generated_audio
    context_audio.fill_(0.0)
    context_latent.fill_(0.0)
    generated_audio.fill_(0.0)
    generated_latent = CAE.encode(generated_audio).unsqueeze(1)
    print("Reset: context_audio, context_latent, generated_latent, and generated_audio zeroed.")
    # print_tensor(True)


def print_tensor(unused_addr, *args):
    """OSC /print — save audio and debug plots to disk."""
    global context_audio, generated_latent, context_latent

    torchaudio.save("context_audio.wav",           context_audio,   44100)
    torchaudio.save("generated_audio.wav", generated_audio, 44100)
    print(f"context_audio.wav ({context_audio.shape}) and audio_generated.wav ({generated_audio.shape})")

    track_names = ["Bass", "Drums", "Guitar", "Piano"]

    # Input mixture plot
    plt.figure(figsize=(12, 10))
    for track_id in range(context_audio.size(0)):
        plt.subplot(context_audio.size(0), 1, track_id + 1)
        plt.plot(context_audio[track_id].cpu().numpy(), label=track_names[track_id])
        plt.title(f"{track_names[track_id]} Audio Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig("context_audio.png")
    plt.close()

    # Generated audio plot
    plt.figure(figsize=(12, 10))
    for track_id in range(generated_audio.size(0)):
        plt.subplot(generated_audio.size(0), 1, track_id + 1)
        plt.plot(generated_audio[track_id].cpu().numpy(), label=track_names[track_id])
        plt.title(f"{track_names[track_id]} Generated")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig("generated_audio.png")
    plt.close()

    # Context latent heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(context_latent[0, 0].cpu().numpy())
    plt.title("Context Latent")
    plt.tight_layout()
    plt.savefig("context_latent.png")
    plt.close()

    # Generated latent heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(generated_latent[0, 0].cpu().numpy())
    plt.title("Generated Latent")
    plt.tight_layout()
    plt.savefig("generated_latent.png")
    plt.close()

    print("Plots saved to disk.")


def packet_test_handler(unused_addr, packet_size, *values):
    """OSC /packet_test — echo back a random float packet of the same size."""
    received_size = len(values)
    print(f"Received test packet with {received_size} floats")
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    msg    = osc_message_builder.OscMessageBuilder(address="/packet_test_response")
    msg.add_arg(received_size)
    for _ in range(received_size):
        msg.add_arg(float(np.random.rand()), arg_type="f")
    bundle.add_content(msg.build())
    client.send(bundle.build())


def update_package_size(unused_addr, new_package_size):
    """OSC /update_package_size — set how many floats are sent per UDP chunk."""
    global package_size
    if new_package_size < 128 or new_package_size > 16384:
        print(f"Invalid package size: {new_package_size} (must be 128–16384)")
        return
    package_size = int(new_package_size)
    print(f"Package size → {package_size}")


def update_r(unused_addr, new_r):
    """OSC /update_r — set the inpainting window as a fraction of buffer length."""
    global r, mask
    if not (0.0 <= new_r <= 1.0):
        print(f"Invalid r: {new_r}")
        return
    r = float(new_r)
    mask = create_temporal_mask(generated_latent, mask_ratio=r).to(device)
    print(f"r → {r}  (mask updated)")


def handle_verbose(unused_addr, state):
    """OSC /verbose [0|1] — enable or disable timing log (mirrored from Max verbose message)."""
    global verbose
    verbose = int(state)
    print(f"Verbose {'on' if verbose else 'off'}")


def update_headroom(unused_addr, ratio):
    """OSC /update_fade — set crossfade overlap as fraction of sampling rate (e.g. 0.02)."""
    global headroom_ratio
    if not (0.0 <= ratio <= 1.0):
        print(f"Invalid fade ratio: {ratio} (must be 0.0–1.0)")
        return
    headroom_ratio = float(ratio)
    print(f"Headroom ratio → {headroom_ratio}")


def update_w(unused_addr, new_w):
    """OSC /w — scale the prediction window relative to inpainting window."""
    global w, mask
    if new_w not in (-1.0, 0.0, 1.0):
        print(f"Invalid w: {new_w} (must be -1, 0, or 1)")
        return
    w = float(new_w)
    print(f"w → {w}")


# =============================================================================
# Raw UDP server — single persistent thread, no per-packet thread spawning
# =============================================================================

_AUDIO_ADDRESSES = {b'/context': 0}

_CONTROL_HANDLERS = {}  # populated in start_server after all handlers are defined


def _osc_read_string(data: bytes, offset: int):
    """Read a null-terminated OSC string and advance offset to next 4-byte boundary."""
    end = data.index(b'\x00', offset)
    s = data[offset:end]
    offset = (end + 4) & ~3
    return s, offset


def _raw_udp_listener(sock):
    """Main receive loop — parses OSC packets directly without spawning threads."""
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            addr_bytes, offset = _osc_read_string(data, 0)

            if addr_bytes in _AUDIO_ADDRESSES:
                if _loading:
                    continue
                # Fast path: skip type tag, unpack 3 ints + floats directly
                _, offset = _osc_read_string(data, offset)
                batch_id, start_index, total_chunks = struct.unpack_from('>iii', data, offset)
                offset += 12
                n = (len(data) - offset) // 4
                values = struct.unpack_from(f'>{n}f', data, offset)
                track_id = _AUDIO_ADDRESSES[addr_bytes]
                buffer_handler(addr_bytes.decode(), (track_id,), batch_id, start_index, total_chunks, *values)
            else:
                # Slow path: use pythonosc to parse args, dispatch to handler
                from pythonosc.osc_message import OscMessage
                msg = OscMessage(data)
                handler = _CONTROL_HANDLERS.get(msg.address)
                if handler:
                    Thread(target=handler, args=(msg.address, *msg), daemon=True).start()
                else:
                    print(f"[UDP] unknown address: {msg.address}")
        except Exception as e:
            print(f"[UDP] error: {e}")


# =============================================================================
# Server startup
# =============================================================================

def start_server(ip, port):
    _CONTROL_HANDLERS.update({
        "/reset":               reset_tensor,
        "/print":               print_tensor,
        "/packet_test":         packet_test_handler,
        "/update_package_size": update_package_size,
        "/update_r":   update_r,
        "/update_fade":         update_headroom,
        "/predict_instruments": handle_predict_instruments,
        "/w":          update_w,
        "/load_model":          load_network,
        "/predict":             predict,
        "/verbose":             handle_verbose,
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((ip, port))
    print(f"\nStarting server on {ip}:{port}  |  device: {device}")
    print("Server is running.\n")
    client.send_message("/ready", True)  # send after bind — socket is now listening
    try:
        _raw_udp_listener(sock)
    except KeyboardInterrupt:
        print("Server stopped.")
        sock.close()


if __name__ == "__main__":
    # Print environment info banner
    _hostname = socket.gethostname()
    try:    _local_ip = socket.gethostbyname(_hostname)
    except: _local_ip = "unknown"
    _gpu_info = (", ".join(torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count()))
                 if torch.cuda.is_available() else "none")
    print("=" * 60)
    print(f"  multi_track server")
    print(f"  host    : {_hostname}")
    print(f"  ip      : {_local_ip}")
    print(f"  os      : {platform.system()} {platform.release()}")
    print(f"  python  : {platform.python_version()}")
    print(f"  gpu     : {_gpu_info}")
    print(f"  pytorch : {torch.__version__}")
    print("=" * 60)

    seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default=None,        help="Compute device (e.g. cuda, cuda:1, cpu)")
    parser.add_argument("--server_ip",  default="0.0.0.0",   help="IP address to listen on")
    parser.add_argument("--client_ip",  default="127.0.0.1", help="IP address of Max/MSP client")
    parser.add_argument("--serverport", type=int, default=7000, help="Server listen port")
    parser.add_argument("--clientport", type=int, default=8000, help="Max/MSP client port")
    args = parser.parse_args()

    if args.device is not None:
        device = args.device

    # UDP client for sending results back to Max
    client = udp_client.SimpleUDPClient(args.client_ip, args.clientport)
    client._sock.setblocking(True)
    client._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    print(f"\nClient: {args.client_ip}:{args.clientport}")

    start_server(args.server_ip, args.serverport)
