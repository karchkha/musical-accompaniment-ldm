"""
multi_track OSC server — CD (Consistency Distillation) model variant.

Receives audio mixture chunks from Max/MSP via OSC, runs inpainting with a
CTM/CD latent model using music2latent CAE, and streams predicted stem
waveforms back in enumerated chunks for reassembly on the Max side.

If launching returns 'Address already in use':
  Run 'sudo lsof -i:7000' (or whichever port) to find the occupying PID,
  then 'kill <PID>' to free it.

Author: Tornike Karchkhadze  tkarchkhadze@ucsd.edu
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import platform
import socket
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
from ctm.utils import EMAAndScales_Initialiser, create_model_and_diffusion_audio
from ctm.enc_dec_lib import load_feature_extractor
from ctm.sample_util import karras_sample
from main.audio_ctm import Audio_CTM_Model

# ── Suppress noisy upstream warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning,  message=".*flash attention.*")

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Global state
# =============================================================================

# Input mixture buffer (1 track × 6 seconds @ 44100 Hz)
tensor        = torch.full((1, 264600), 0.0)

# Latent representation and inpainting mask (updated each predict cycle)
latent = mask = torch.full((1, 1, 64, 64), 0.0)

# Last generated audio — kept for shift_tensor_data and debug export
generated_audio = torch.full((1, 264600), 0.0)

# Model handles (populated by load_network)
latent_diffusion  = None
diffusion_sampler = None
CAE               = EncoderDecoder(device=device)  # music2latent audio codec

# Runtime parameters (can be updated via OSC at any time)
steps        = 2
config       = {}
package_size = 5120   # floats per UDP chunk — tune from Max with /update_package_size
percentage   = 0.25   # fraction of the window to inpaint
pr_win_mul   = 1.0    # prediction window multiplier

stems_to_inpaint   = []
stemidx_to_inpaint = []

verbose = 0   # 0 = silent, 1 = print timing events (toggled via /verbose from Max)

filename = "configs/for_server/CD_latent_cond_gen_concat_inpaint.yaml"

# Batch placeholder passed to model helpers
batch = [
    torch.full((1, 1, 264600), 0.0),
    torch.zeros(1, 4),
    torch.zeros((1, 4, 264600)),
]

# Per-stem output waveform cache (populated during predict)
waveforms = {"bass": None, "drums": None, "guitar": None, "piano": None}


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

def _make_osc_dgram(address: str, chunk_idx: int, total_chunks: int,
                    floats_chunk: np.ndarray) -> bytes:
    """Build a raw OSC datagram: /address [chunk_idx:i] [total_chunks:i] [floats:f...].

    ~100x faster than OscMessageBuilder: uses numpy byte packing instead of
    a Python loop with per-element add_arg() calls.
    """
    import struct as _struct
    addr = address.encode() + b'\x00'
    addr += b'\x00' * ((4 - len(addr) % 4) % 4)
    n    = len(floats_chunk)
    tag  = (',ii' + 'f' * n).encode() + b'\x00'
    tag += b'\x00' * ((4 - len(tag) % 4) % 4)
    header = _struct.pack('>ii', chunk_idx, total_chunks)
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
    """Create a mask that covers the last `mask_ratio` fraction of the time axis.

    Shape: same as `like` (B, C, F, T).  False = masked region to inpaint.
    """
    _, _, F, T = like.shape
    mask = torch.ones_like(like)
    t_mask  = int(T * mask_ratio)
    t_start = T - t_mask
    mask[:, :, :, t_start:] = False
    return mask


# =============================================================================
# Model loading
# =============================================================================

def load_network(unused_addr):
    """OSC handler for /load_model — loads CTM/CD checkpoint and prepares for inference."""
    global latent_diffusion, diffusion_sampler, stemidx_to_inpaint, steps
    global tensor, latent, mask, config, package_size, filename
    global percentage, pr_win_mul, CAE

    config = yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
    cfg    = dict2namespace(config)

    # Instantiate CTM model wrapper
    model = Audio_CTM_Model(cfg)
    print("\nCD model instantiated.")

    # Load checkpoint weights
    checkpoint_path = cfg.resume_from_checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint path — running with random initialization.")

    # Extract net and sampler, then discard the wrapper
    latent_diffusion, diffusion_sampler = model.net, model.diffusion
    del model

    latent_diffusion.to(device)
    latent_diffusion.eval()

    steps = cfg.audio_samples_logger.steps_to_calculate_metrics

    # Pre-compute inpainting mask and initial latent encoding
    mask   = create_temporal_mask(mask, mask_ratio=percentage).to(device)
    latent = CAE.encode(tensor).unsqueeze(1)

    print("Model ready!\n")


# =============================================================================
# Prediction
# =============================================================================

def predict(*args):
    """Run one CD inpainting prediction cycle and stream results back to Max via OSC."""
    global latent_diffusion, diffusion_sampler, tensor, waveforms
    global stems_to_inpaint, stemidx_to_inpaint, steps, batch
    global package_size, percentage, config, latent, CAE
    global _first_chunk_time, _last_chunk_time, _chunk_count

    t_predict_received = time.time()

    # Guard against concurrent predictions
    if not _predict_sem.acquire(blocking=False):
        print(f"[PREDICT] skipped — already running  {_hms()}")
        return

    if verbose:
        gap_since_first = (t_predict_received - _first_chunk_time) if _first_chunk_time else -1
        gap_since_last  = (t_predict_received - _last_chunk_time)  if _last_chunk_time  else -1
        print("----------------------------------------")
        print(f"[PREDICT] triggered  {_chunk_count} chunks  "
              f"first {gap_since_first*1000:.1f}ms ago  last {gap_since_last*1000:.1f}ms ago  {_hms()}")

    # Drain the incoming data queue before touching the tensor
    message_queue.join()
    if verbose:
        print(f"  queue drained  +{(time.time() - t_predict_received)*1000:.1f}ms  {_hms()}")

    timer.checkpoints.clear()
    if verbose:
        timer.record_event("Predict start")

    with torch.no_grad():

        # One-hot feature vector: 1 = preserved stem, 0 = generated stem
        current_features = torch.zeros(
            1, len(config['audio_samples_logger']['stems']), device=device)
        for idx in stemidx_to_inpaint:
            current_features[:, idx] = 1

        # Encode mixture and zero out the prediction window
        mixture_latent = CAE.encode(tensor).unsqueeze(1)
        if verbose: timer.record_event("Mixture latent encoded")

        start_idx = int(mixture_latent.size(-1) * (1 - pr_win_mul * percentage))
        mixture_latent[:, :, :, start_idx:] = 0.0

        # Pass source latent and mask into the sampler via model_kwargs
        source = latent.clone()
        model_kwargs = {
            "class_labels":    current_features,
            "augment_labels":  None,
            "mixture":         mixture_latent,
            "source":          source,
            "mask":            mask,
        }

        sampler = config['audio_samples_logger']['sampler_to_calculate_metrics']
        if verbose: timer.record_event("Entering sampler")

        samples = karras_sample(
            diffusion=diffusion_sampler,
            model=latent_diffusion,
            shape=(1,
                   config['audio_samples_logger']['channels'],
                   config['model']['img_resolution'],
                   config['model']['img_resolution']),
            steps=steps,
            model_kwargs=model_kwargs,
            device=device,
            clip_denoised=False,
            sampler=sampler,
            generator=None,
            teacher=False,
            ctm=False,
            x_T=None,
            clip_output=False,
            sigma_min=config['diffusion']['sigma_min'],
            sigma_max=config['diffusion']['sigma_max'],
            train=False,
        )
        if verbose: timer.record_event("Sampling done")

        # Update stored latent with the freshly inpainted region
        start_idx = int(samples.size(-1) * (1 - percentage))
        latent[:, :, :, start_idx:] = samples[:, :, :, start_idx:].clone()

        # Decode only the cols needed for the send window + 1 col of headroom.
        # n_cols scales with percentage so it stays correct if percentage changes via OSC.
        total_length     = config['audio_samples_logger']['length']
        headroom_samples = int(0.02 * config['audio_samples_logger']['sampling_rate'])
        n_needed         = int(total_length * percentage) + headroom_samples

        n_cols           = int(64 * percentage) + 1   # e.g. 0.25 → 17, 0.5 → 33
        samples_per_col  = total_length / 64           # 4134.375 — architectural constant
        expected_len     = int(n_cols * samples_per_col)

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
        stem_names = ["bass", "drums", "guitar", "piano"]
        dest       = (client._address, client._port)

        for i, stem_name in enumerate(stem_names):
            if i not in stemidx_to_inpaint:
                continue

            if verbose: print(f"  [SEND]    {stem_name}  {_hms()}")
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
                    _make_osc_dgram("/" + stem_name, chunk_idx, total_chunks, chunk),
                    dest)

    if verbose: timer.record_event("Send complete")

    client.send_message("/server_predicted", True)

    # Slide all buffers forward by one prediction window
    shift_size = int(tensor.size(-1) * percentage)
    shift_tensor_data(tensor,          percentage)
    shift_tensor_data(latent,          percentage)
    shift_tensor_data(generated_audio, percentage)
    if verbose:
        print(f"  Buffers shifted left by {shift_size} samples.  {_hms()}")
        timer.record_event("Buffers shifted")
    _predict_sem.release()


# =============================================================================
# Incoming data queue — buffers OSC chunks into the mixture tensor
# =============================================================================

message_queue = Queue()


def process_message_queue():
    """Worker thread: dequeues incoming audio chunks and writes them into `tensor`."""
    global tensor, config, percentage, pr_win_mul
    while True:
        track_id, start_index, values = message_queue.get()

        depth       = tensor.size(-1)
        # Target write window: [100 - (pr_win_mul+1)*pct, 100 - pct] of the buffer
        start_idx   = int(depth * (1 - (pr_win_mul + 1) * percentage))
        range_start = start_idx + start_index
        range_end   = range_start + len(values)

        track_id = 0  # single-track mode — mixture is always track 0
        tensor[track_id, range_start:range_end] = torch.tensor(values)

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
_predict_sem          = Lock()          # prevents concurrent predictions
_watchdog_timer       = None
_WATCHDOG_SAFETY_FACTOR = 5             # fire after 5× the observed avg inter-chunk gap


def _watchdog_fire(batch_id):
    """Called when chunks stop arriving before the expected count is reached."""
    global _auto_chunks_received
    with _auto_lock:
        if _current_batch_id != batch_id or _auto_chunks_received == 0:
            return  # batch already completed or superseded
        missing = _auto_chunks_expected - _auto_chunks_received
        print(f"WARNING: batch {batch_id} missing {missing}/{_auto_chunks_expected} "
              f"chunks — triggering predict anyway")
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
    global _auto_chunks_received, _auto_chunks_expected, _current_batch_id, _watchdog_timer

    t                     = time.time()
    track_id              = int(track_id[0])
    start_index           = int(start_index)
    batch_id              = int(batch_id)
    total_expected_chunks = int(total_expected_chunks)

    message_queue.put((track_id, start_index, values))

    with _auto_lock:
        if batch_id != _current_batch_id:
            _current_batch_id     = batch_id
            _auto_chunks_received = 0
            _first_chunk_time     = t
            _chunk_count          = 0
            if verbose:
                print(f"[RX]      first chunk  batch={batch_id}  "
                      f"expecting {total_expected_chunks} chunks  {_hms()}")

        _auto_chunks_expected  = total_expected_chunks
        _auto_chunks_received += 1
        _chunk_count          += 1
        _last_chunk_time       = t
        should_predict         = (_auto_chunks_received >= _auto_chunks_expected)

        if should_predict:
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

def shift_tensor_data(tensor: torch.Tensor, percentage: float):
    """Shift the last dimension left by `percentage`, zero-filling the tail.

    Used to slide the rolling mixture buffer and latent forward after each
    prediction cycle so the next window is positioned correctly.
    """
    if not (0.0 <= percentage <= 1.0):
        raise ValueError(f"percentage must be 0–1, got {percentage}")

    shift_size = int(tensor.size(-1) * percentage)
    if shift_size == 0:
        return

    temp = tensor.clone()
    tensor[..., :-shift_size] = temp[..., shift_size:]
    tensor[..., -shift_size:] = 0.0


# =============================================================================
# OSC handlers — control messages
# =============================================================================

def handle_predict_instruments(address, *args):
    """OSC /predict_instruments [bass drum guitar piano] — 1 = generate, 0 = preserve."""
    global stemidx_to_inpaint
    if len(args) != 4:
        print(f"Invalid /predict_instruments message: {args}")
        return
    instrument_names   = ["bass", "drums", "guitar", "piano"]
    stems_to_inpaint   = [instrument_names[i] for i in range(4) if args[i] == 1]
    stemidx_to_inpaint = [i for i, s in enumerate(instrument_names)
                           if s in stems_to_inpaint]
    print(f"Stems to generate: {stems_to_inpaint}")


def reset_tensor(unused_addr, *args):
    """OSC /reset — zero out the mixture buffer."""
    global tensor
    tensor.fill_(0.0)
    print("Tensor reset to 0.0")
    print_tensor(True)


def print_tensor(unused_addr, *args):
    """OSC /print — save audio and debug plots to disk."""
    global tensor, latent

    torchaudio.save("audio.wav",           tensor,          44100)
    torchaudio.save("audio_generated.wav", generated_audio, 44100)
    print(f"Saved audio.wav ({tensor.shape}) and audio_generated.wav ({generated_audio.shape})")

    track_names = ["Bass", "Drums", "Guitar", "Piano"]

    # Input mixture plot
    plt.figure(figsize=(12, 10))
    for track_id in range(tensor.size(0)):
        plt.subplot(tensor.size(0), 1, track_id + 1)
        plt.plot(tensor[track_id].cpu().numpy(), label=track_names[track_id])
        plt.title(f"{track_names[track_id]} Audio Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig("tensor_plot_subplots.png")
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
    plt.savefig("tensor_plot_subplots_generated.png")
    plt.close()

    # Latent heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(latent[0, 0].cpu().numpy())
    plt.title("Latent")
    plt.tight_layout()
    plt.savefig("latent.png")
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


def update_percentage(unused_addr, new_percentage):
    """OSC /update_percentage — set the inpainting window as a fraction of buffer length."""
    global percentage
    if not (0.0 <= new_percentage <= 1.0):
        print(f"Invalid percentage: {new_percentage}")
        return
    percentage = float(new_percentage)
    print(f"Percentage → {percentage}")


def handle_verbose(unused_addr, state):
    """OSC /verbose [0|1] — enable or disable timing log (mirrored from Max verbose message)."""
    global verbose
    verbose = int(state)
    print(f"Verbose {'on' if verbose else 'off'}")


def update_pr_win_mul(unused_addr, new_pr_win_mul):
    """OSC /pr_win_mul — scale the prediction window relative to inpainting window."""
    global pr_win_mul, mask
    if not (0.0 <= new_pr_win_mul <= 2.0):
        print(f"Invalid pr_win_mul: {new_pr_win_mul} (must be 0–2)")
        return
    pr_win_mul = float(new_pr_win_mul)
    print(f"pr_win_mul → {pr_win_mul}")


# =============================================================================
# OSC dispatcher registration
# =============================================================================

dispatcher = dispatcher.Dispatcher()

# Incoming audio chunks — track_id is passed as extra arg by dispatcher.map
dispatcher.map("/bass",   buffer_handler, 0)
dispatcher.map("/drums",  buffer_handler, 1)
dispatcher.map("/guitar", buffer_handler, 2)
dispatcher.map("/piano",  buffer_handler, 3)

# Control messages
dispatcher.map("/reset",               reset_tensor)
dispatcher.map("/print",               print_tensor)
dispatcher.map("/packet_test",         packet_test_handler)
dispatcher.map("/update_package_size", update_package_size)
dispatcher.map("/update_percentage",   update_percentage)
dispatcher.map("/predict_instruments", handle_predict_instruments)
dispatcher.map("/pr_win_mul",          update_pr_win_mul)
dispatcher.map("/load_model",          load_network)
dispatcher.map("/predict",             predict)
dispatcher.map("/verbose",             handle_verbose)


# =============================================================================
# Server startup
# =============================================================================

def start_server(ip, port):
    print(f"\nStarting server on {ip}:{port}  |  device: {device}")
    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    server.max_packet_size = 65536
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    print("Server is running.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    seed_everything(1234)

    # Print environment info banner
    _hostname = socket.gethostname()
    try:    _local_ip = socket.gethostbyname(_hostname)
    except: _local_ip = "unknown"
    _gpu_info = (", ".join(torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count()))
                 if torch.cuda.is_available() else "none")
    print("=" * 60)
    print(f"  multi_track server  [CD]")
    print(f"  host    : {_hostname}")
    print(f"  ip      : {_local_ip}")
    print(f"  os      : {platform.system()} {platform.release()}")
    print(f"  python  : {platform.python_version()}")
    print(f"  gpu     : {_gpu_info}")
    print(f"  pytorch : {torch.__version__}")
    print("=" * 60)

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

    client.send_message("/ready", True)

    start_server(args.server_ip, args.serverport)
