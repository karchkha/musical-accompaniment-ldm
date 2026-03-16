"""
If executing this script returns an 'Address already in use' error
make sure there are no processes running on the ports already.
To do that run 'sudo lsof -i:9997' 'sudo lsof -i:9998'
(9997 and 9998 are the default ports used here, so adjust accordingly
if using different ports) This commands brings up list of processes using these ports,
and gives their PID. For each process type, 'kill XXXX' where XXXX is PID.
"""


import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder

from threading import Thread

import sys
sys.path.append("src")
from pytorch_lightning import seed_everything
import yaml
# from latent_diffusion.models.musicldm import MusicLDM, DDPM, DDIMSampler
from main.module_base import Audio_LDM_Model
import importlib
import torch.nn.functional as F
# import src.utilities.audio as Audio
import torchaudio
# from torchinfo import summary
import time

# import threading

from queue import Queue

# import time



# import pickle
# import numpy as np
# from sklearn.decomposition import sparse_encode


#import multiprocessing

######################################################################################################################################################################################################################################

# Create a tensor of size [4, 163840] initialized with -inf
# tensor = torch.full((4, 163840), float('-inf'))

tensor = torch.full((1, 264600), 0.0)

latent = mask = torch.full((1, 1, 64, 64), 0.0)

generated_audio = torch.full((1, 264600), 0.0)

latent_diffusion = None  ### Network
MSAProc = None           ### Audio processing
stems_to_inpaint = []
stemidx_to_inpaint = []
steps = 10 
config = {}
package_size = 5120
percentage = 0.25
pr_win_mul = 1.0
filename = "configs/for_server/Diff_latent_cond_gen_concat_eval.yaml"
diffusion_sampler = None
diffusion_schedule = None

batch = [
    torch.full((1, 1, 264600), 0.0),  # Tensor filled with -12.00
    torch.zeros(1, 4),  # Tensor filled with zeros
    torch.zeros((1, 4, 264600)),  # Tensor filled with zeros
]

# Initialize empty placeholders for output waveforms
waveforms = {
    "bass": None,
    "drums": None,
    "guitar": None,
    "piano": None
}


class EventTimer:
    def __init__(self):
        self.checkpoints = []

    def record_event(self, event_name="Event"):
        """Records the event with the current timestamp and reports time difference from the previous event."""
        current_time = time.time()
        if self.checkpoints:
            prev_time = self.checkpoints[-1][1]
            time_diff = current_time - prev_time
            print(f"{event_name}: {time_diff:.4f} Sec")
        else:
            print(f"{event_name} recorded. This is the first event.")
        self.checkpoints.append((event_name, current_time))

    def get_intervals(self):
        """Returns a list of time intervals between consecutive events."""
        return [(self.checkpoints[i][0], self.checkpoints[i][1] - self.checkpoints[i - 1][1])
                for i in range(1, len(self.checkpoints))]

timer = EventTimer()

# # Function to process a single OSC message in a separate thread
# def process_message(track_id, start_index, values):
#     global tensor

#     # start_time = time.time()  # Start timing for import
#     # formatted_start_time = time.strftime("%M:%S", time.localtime(start_time)) + f".{int(start_time * 1000) % 1000:03d}"
#     # print(f"Processing started at: {formatted_start_time}")
 
#     # Ensure track_id is valid
#     if track_id < 0 or track_id >= tensor.size(0):
#         print(f"Invalid track_id: {track_id}. Skipping.")
#         return

#     # Ensure start_index is within bounds
#     if start_index < 0 or start_index >= tensor.size(1):
#         print(f"Invalid start_index: {start_index}. Skipping.")
#         return

#     # Compute the range of indices to fill
#     end_index = start_index + len(values)
#     if end_index > tensor.size(1):
#         print(f"End index {end_index} exceeds tensor bounds. Clipping to {tensor.size(1)}.")
#         end_index = tensor.size(1)

#     # Fill the tensor with the received values
#     num_values = end_index - start_index
#     tensor[track_id, start_index:end_index] = torch.tensor(values[:num_values])

#     # Print debug information
#     # print(f"Track {track_id} - Received {len(values)} values. Filled indices {start_index} to {end_index}.")
    
#     # end_time = time.time()  # End timing for import
#     # formatted_end_time = time.strftime("%M:%S", time.localtime(end_time)) + f".{int(end_time * 1000) % 1000:03d}"
#     # print(f"Processing ended at: {formatted_end_time}")
    
#     # elapsed_time = end_time - start_time
#     # print(f"Track {track_id} - Import completed in {elapsed_time:.6f} seconds.")
#     # print(f"Track {track_id} - Received {len(values)} values. Filled indices {start_index} to {end_index}.")

# # Updated buffer_handler to spawn threads
# def buffer_handler(unused_addr, track_id, start_index, *values):
    
#     track_id = int(track_id[0])  # Ensure track_id is an integer
#     start_index = int(start_index)  # Ensure start_index is an integer

#     # Start a new thread to handle the message
#     thread = Thread(target=process_message, args=(track_id, start_index, values))
#     thread.start()

##################################################################################################################################################################################

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
    
    module_path, class_name = config['_target_'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    
    # Remove _target_ from the config dictionary and instantiate the object
    config_dict = {k: v for k, v in config.items() if k != '_target_'}
    return cls(**config_dict, **kwargs)

def load_network(unused_addr):
    global latent_diffusion, stemidx_to_inpaint, steps, tensor, MSAProc, latent, mask, config, package_size, filename, percentage, diffusion_sampler, diffusion_schedule, pr_win_mul
           
    config = yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
       
    cfg = dict2namespace(config)
    
    # Init Model
    diffusion_sigma_distribution = instantiate_from_config(cfg.diffusion_sigma_distribution)
    latent_diffusion = instantiate_from_config(cfg.model, diffusion_sigma_distribution = diffusion_sigma_distribution)
    print("\nStarted Duffusion model!")
    
    # Ensure checkpoint path is defined
    checkpoint_path = cfg.resume_from_checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load safely on CPU
        latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)  # Load model weights
        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint path provided. Running model with random initialization.")

    latent_diffusion.to("cuda:0")

    ### Init Sampler
    diffusion_sampler = instantiate_from_config(cfg.diffusion_sampler)
    diffusion_schedule = instantiate_from_config(cfg.diffusion_schedule)

    steps=cfg.audio_samples_logger.sampling_steps

    # MSAProc = MultiSourceAudioProcessor(cfg)
    mask = create_temporal_mask(mask, mask_ratio = percentage).to("cuda:0")
    
    latent = latent_diffusion.CAE.encode(tensor).unsqueeze(1) 

    # client.send_message("/ready", True)
    print("Model is ready!")

#########################################################################


#########################################################################

def create_temporal_mask(like, mask_ratio):
    """
    Creates a temporal mask over the image-like spectrogram.
    - mask_ratio: percentage of the time axis to mask (e.g., 50%)
    - Assumes: First dim = Frequency (F), Second dim = Time (T)
    """
    _, _, F, T = like.shape  # Batch, Channels, Frequency, Time
    device = like.device
    mask = torch.ones_like(like, dtype=torch.bool)

    # Compute time range to mask (masking the last portion)
    t_mask = int(T * mask_ratio)  # Number of time steps to mask
    t_start = T - t_mask  # Start masking from this index

    # Apply mask to the last portion of the time axis
    mask[:, :, :, t_start:] = False  # Set masked area to False
    return mask

#########################################################################

def predict(*args):
    global latent_diffusion, tensor, waveforms, stems_to_inpaint, stemidx_to_inpaint, steps, batch, MSAProc, package_size, percentage, config, diffusion_sampler, diffusion_schedule, z, config, latent
    
    # batch = MSAProc.fill_batch_from_audio(tensor, batch)
    timer.record_event("\nStart pred. function")  # First event

    with torch.no_grad():

        noise = torch.randn(
            (1, config['audio_samples_logger']['channels'], latent_diffusion.model.diffusion.net.img_resolution, latent_diffusion.model.diffusion.net.img_resolution), device=latent_diffusion.device
        )

       # Create a zero-initialized feature tensor for batch size 1
        current_features = torch.zeros(1, len(config['audio_samples_logger']['stems']), device=latent_diffusion.device)

        # Set the one-hot vector for preserving the given stems
        for idx in stemidx_to_inpaint:
            current_features[:, idx] = 1  # Mark preserved stems        
        
        # latent, class_indexes, mixture_latent, embedding, mixture_features_channels_list = latent_diffusion.get_input(batch, current_features)
        ###########################
  
        mixture_latent = latent_diffusion.CAE.encode(tensor).unsqueeze(1)
        
        timer.record_event("Calculated Mixrute latent")  # First event
        
        ## Add pr_win_mul * percentage noise patch to the mixture to make future prediction possible
        start_idx = int(mixture_latent.size(-1)  * (1 - pr_win_mul*percentage))
        mixture_latent[:, :, :, start_idx:] = 0.0 # noise[:, :, :, start_idx:].clone()
                    
        ###########################

        # Mask part of the image
        inpaint = latent.clone()
        # Inject noise in the masked area
        inpaint = torch.where(mask, inpaint, noise.to(inpaint.dtype))
        
        timer.record_event("Entering the Sampler")  # First event

        # Inpaint from the model using the noise and the current one-hot features
        samples = latent_diffusion.model.inpaint(
            inpaint=inpaint,
            inpaint_mask = mask, 
            noise_labels_s=None,
            # features=current_features,
            sampler=diffusion_sampler,
            sigma_schedule=diffusion_schedule,
            num_steps=steps,
            class_labels = current_features,
            augment_labels=None,
            mixture=mixture_latent,
            # channels_list=channels_list,
            # mixture_features_channels_list=mixture_features_channels_list,
        )
        timer.record_event("Done Sampling")  # First event
        
        # update latet vectro for future inpainting
        start_idx = int(samples.size(-1)  * (1 - percentage))
        latent[:, :, :, start_idx:] = samples[:, :, :, start_idx:].clone()
        
        samples_wav = latent_diffusion.CAE.decode(samples.squeeze(1)).unsqueeze(1)
        timer.record_event("Converted to wav")  # First event
        
        # samples_wav = F.pad(samples_wav, (0, config['audio_samples_logger']['length'] - samples_wav.size(-1)), mode="constant", value=0).cpu().numpy()
        actual_length = samples_wav.size(-1)
        samples_wav = samples_wav.cpu().numpy()
        
        # Add 2 milliseconds of headroom at the beginning
        headroom_samples = int(0.02 * config['audio_samples_logger']['sampling_rate'])  # Convert ms to samples
        fade_in_window = np.linspace(0, 1, headroom_samples)
        
        timer.record_event("Starting sending")  # First event
        
        # Fill the waveforms for stems in stemidx_to_inpaint
        stem_names = ["bass", "drums", "guitar", "piano"]
        for i in range(4):  # Loop through indices 0 to 3
            if i in stemidx_to_inpaint:
                stem_name = stem_names[i]
                waveform = samples_wav
                print(f"Generated waveform for {stem_name}")

                # Flatten and convert to float32
                flatten_prediction = waveform.flatten().astype(np.float32)

                # Calculate the range to send (last percentage part)
                total_length = config['audio_samples_logger']['length']
                start_idx = max(0, int(total_length * (1 - percentage)) - (total_length - actual_length) - headroom_samples)         
                end_idx = actual_length #config['audio_samples_logger']['length']

                flatten_prediction[start_idx:start_idx + headroom_samples] *= fade_in_window
                
                generated_audio[:,start_idx:end_idx] = torch.tensor(flatten_prediction[start_idx:end_idx])   #### this will be deleted
                
                # for i in range(0, 163840, package_size):
                # Send only the percentage part in packages
                for j in range(start_idx, end_idx, package_size):
                    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
                    msg = osc_message_builder.OscMessageBuilder(address="/"+stem_name)

                    # Determine the actual number of elements in this package
                    remaining = min(package_size, end_idx - j)  # Ensures last package is correct

                    # Add the correct amount of data
                    for k in range(remaining):  
                        prediction_for_sending = flatten_prediction[j + k]
                        msg.add_arg(prediction_for_sending, arg_type="f")  # Specify 'f' for float32

                    # Add the message to the bundle
                    bundle.add_content(msg.build())
                    bundle = bundle.build()

                    # Send the bundle
                    client.send(bundle)

                    # time.sleep(0.0001)
     
    timer.record_event("Done sending")  # First event
                        
    client.send_message("/server_predicted", True)

    # Shift tensor data by percentage
    # percentage = config['data']['params']['path']['percentage']
    shift_tensor_data(tensor, percentage)
    shift_tensor_data(latent, percentage)
    shift_tensor_data(generated_audio, percentage)

    timer.record_event("Done shifting")  # First event
    
# Create a queue to hold incoming messages
message_queue = Queue()

# # Function to process messages from the queue
# def process_message_queue():
#     global tensor
#     while True:
#         # Get a message from the queue
#         track_id, start_index, values = message_queue.get()

#         # # Process the message
#         # track_id = track_id
#         # if track_id < 0 or track_id >= tensor.size(0):
#         #     print(f"Invalid track_id: {track_id}. Skipping.")
#         #     continue

#         # if start_index < 0 or start_index >= tensor.size(1):
#         #     print(f"Invalid start_index: {start_index}. Skipping.")
#         #     continue

#         end_index = start_index + len(values)
#         # if end_index > tensor.size(1):
#         #     print(f"End index {end_index} exceeds tensor bounds. Clipping to {tensor.size(1)}.")
#         #     end_index = tensor.size(1)

#         num_values = end_index - start_index
#         tensor[track_id, start_index:end_index] = torch.tensor(values[:num_values])

#         # print(f"Track {track_id} - Received {len(values)} values. Filled indices {start_index} to {end_index}.")

#         # if track_id == 3 and end_index == 163840:
#         #     predict(None)
        
#         # Mark the task as done
#         message_queue.task_done()

def process_message_queue():
    global tensor, config, percentage, pr_win_mul
    while True:
        # Get a message from the queue
        track_id, start_index, values = message_queue.get()

        depth = tensor.size(-1)  # Depth of the tensor

        # Calculate the target range [100 - 2 * percentage, 100 - percentage]
        start_idx = int(depth * (1 - (pr_win_mul+1) * percentage))
        end_idx = int(depth * (1 - percentage))

        # Calculate the chunk's target range
        package_size = len(values)
        range_start = start_idx + start_index
        range_end = range_start + package_size

        track_id = 0 ### manually settting this because we only have 1 track here
        # Populate the tensor with incoming data
        tensor[track_id, range_start:range_end] = torch.tensor(values)

        # print(f"Track {track_id}: Populated indices {range_start} to {range_end}.")

        # Mark the task as done
        message_queue.task_done()


# Start a pool of worker threads
num_workers = 64  # Adjust based on your system's capabilities
for _ in range(num_workers):
    worker = Thread(target=process_message_queue, daemon=True)
    worker.start()

# Updated buffer_handler to enqueue messages
def buffer_handler(unused_addr, track_id, start_index, *values):
    track_id = int(track_id[0])
    start_index = int(start_index)
    # print(track_id, start_index)
    message_queue.put((track_id, start_index, values))



def shift_tensor_data(tensor: torch.Tensor, percentage: float):
    """
    Shifts the data in the last dimension to the left by a given percentage.
    The emptied space at the end is filled with zeros.

    Parameters:
    - tensor (torch.Tensor): Input tensor of any shape.
    - percentage (float): Fraction of the last dimension to shift (0.0 - 1.0).

    Supports tensors of any shape (2D, 3D, 4D, etc.).
    """
    if not (0.0 <= percentage <= 1.0):
        raise ValueError(f"Percentage must be between 0.0 and 1.0, got {percentage}")

    # Get the size of the last dimension
    last_dim_size = tensor.size(-1)
    
    # Calculate the shift size
    shift_size = int(last_dim_size * percentage)
    if shift_size == 0:
        return  # No need to shift if percentage is too small

    # Create a temporary copy to prevent data overlap issues
    temp_tensor = tensor.clone()

    # Shift data left along the last dimension
    tensor[..., :-shift_size] = temp_tensor[..., shift_size:]

    # Zero out the remaining part at the end
    tensor[..., -shift_size:] = 0.0

    print(f"Shifted tensor left by {shift_size} elements along last dimension.")


    # print(f"Tensor data shifted to the left by {shift_size} samples.")




def handle_predict_instruments(address, *args):
    """Handles incoming /predict_instruments OSC messages from Max/MSP."""
    global stemidx_to_inpaint

    if len(args) != 4:
        print(f"Received invalid /predict_instruments message: {args}")
        return

    # Map the incoming values (1 = predict, 0 = send) to instrument names
    instrument_names = ["bass", "drums", "guitar", "piano"]
    stems_to_inpaint = [instrument_names[i] for i in range(4) if args[i] == 1]

    # stems_to_inpaint = config["data"]["params"].get('path', {}).get('stems_to_inpaint', None)
    stems = instrument_names
    stemidx_to_inpaint = [i for i, s in enumerate(stems) if s in stems_to_inpaint]

    print(f"Instruments to be predicted: {stems_to_inpaint}")



# Function to reset the tensor
def reset_tensor(unused_addr, *args):
    global tensor
    tensor.fill_(0.0) #float('-inf'))
    print("Tensor reset to 0.0")
    print_tensor(True)

def print_tensor(unused_addr, *args):
    global tensor, latent

    print(f"Received /print message with args: {args}")

    # Save as WAV file
    torchaudio.save("audio.wav", tensor, 44100)
    print(f"Saved audio.wav with shape {tensor.shape} at 44100 Hz")

    # Save as WAV file
    torchaudio.save("audio_generated.wav", generated_audio, 44100)
    print(f"Saved audio.wav with shape {generated_audio.shape} at 44100 Hz")    
    

    # Plot each track in separate subplots
    track_names = ["Bass", "Drums", "Guitar", "Piano"]
    num_tracks = tensor.size(0)

    plt.figure(figsize=(12, 10))  # Adjust figure size for better readability

    for track_id in range(num_tracks):
        plt.subplot(num_tracks, 1, track_id + 1)  # Create a subplot for each track
        plt.plot(tensor[track_id].cpu().numpy(), label=track_names[track_id])
        plt.title(f"{track_names[track_id]} Audio Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("tensor_plot_subplots.png")
    plt.close()  # Close the figure to free memory
        
    # Plot each track in separate subplots
    track_names = ["Bass", "Drums", "Guitar", "Piano"]
    num_tracks = generated_audio.size(0)

    plt.figure(figsize=(12, 10))  # Adjust figure size for better readability

    for track_id in range(num_tracks):
        plt.subplot(num_tracks, 1, track_id + 1)  # Create a subplot for each track
        plt.plot(generated_audio[track_id].cpu().numpy(), label=track_names[track_id])
        plt.title(f"{track_names[track_id]} Audio Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("tensor_plot_subplots_generated.png")
    plt.close()  # Close the figure to free memory dddd
    
    plt.figure(figsize=(10, 10))  # Adjust figure size for better readability

    plt.subplot(num_tracks, 1, track_id + 1)  # Create a subplot for each track
    plt.imshow(latent[0,0].cpu().numpy())
    plt.title(f"latent")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")
    # plt.grid()
    # plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("latent.png")
    plt.close()  # Close the figure to free memory


    print("Tensor plot with subplots saved to tensor_plot_subplots.png")


def packet_test_handler(unused_addr, packet_size, *values):
    received_size = len(values)
    print(f"Received test packet with {received_size} floats")

    # Generate a response packet of the same size
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    msg = osc_message_builder.OscMessageBuilder(address="/packet_test_response")
    msg.add_arg(received_size)

    for _ in range(received_size):
        msg.add_arg(float(np.random.rand()), arg_type="f")  # Random float

    bundle.add_content(msg.build())
    bundle = bundle.build()
    client.send(bundle)

def update_package_size(unused_addr, new_package_size):
    global package_size


    if new_package_size < 128 or new_package_size > 16384:
        print(f"Invalid package size received: {new_package_size}. Ignoring.")
        return

    package_size = int(new_package_size)
    print(f"Updated package size to {new_package_size}")

def update_percentage(unused_addr, new_percentage):
    global percentage

    if new_percentage < 0.0 or new_percentage > 1.0:
        print(f"Invalid percentage received: {new_percentage}. Ignoring.")
        return

    percentage = float(new_percentage)
    print(f"Updated percentage to {new_percentage}")


def update_pr_win_mul(unused_addr, new_pr_win_mul):
    global pr_win_mul, mask

    if new_pr_win_mul < 0.0 or new_pr_win_mul > 2.0:
        print(f"Invalid pr_win_mul received: {new_pr_win_mul}. Ignoring.")
        return

    pr_win_mul = float(new_pr_win_mul)
    print(f"Updated pr_win_mul to {new_pr_win_mul}")


# Dispatcher to route messages to handlers
dispatcher = dispatcher.Dispatcher()

# Define OSC address mapping
dispatcher.map("/bass", buffer_handler, 0)   # Bass corresponds to track_id 0
dispatcher.map("/drums", buffer_handler, 1)  # Drums corresponds to track_id 1
dispatcher.map("/guitar", buffer_handler, 2) # Guitar corresponds to track_id 2
dispatcher.map("/piano", buffer_handler, 3)  # Piano corresponds to track_id 3
dispatcher.map("/reset", reset_tensor)       # Reset tensor
dispatcher.map("/print", print_tensor)       # Print tensor as a plot
dispatcher.map("/packet_test", packet_test_handler)
dispatcher.map("/update_package_size", update_package_size)
dispatcher.map("/update_percentage", update_percentage)
dispatcher.map("/predict_instruments", handle_predict_instruments)
dispatcher.map("/pr_win_mul", update_pr_win_mul)


dispatcher.map("/load_model", load_network)
dispatcher.map("/predict", predict)

# dispatcher.map("/shift", shift_tensor_data)

# OSC Server Setup
def start_server(ip, port):
    print(f"\nStarting server on {ip}:{port}")
    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    server.max_packet_size = 65536
    print("Server is running!\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")

if __name__ == "__main__":
    seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", default="0.0.0.0", help="The IP address to listen on")
    parser.add_argument("--client_ip", default="127.0.0.1", help="The IP address of client. 127.0.0.1 if the cilent and server are on the same device")
    parser.add_argument("--serverport", type=int, default=7000, help="The port to listen on")
    parser.add_argument("--clientport", type=int, default=8000, help="The client port")
    args = parser.parse_args()

    ### client
    client_port=str(args.clientport)
    client = udp_client.SimpleUDPClient(args.client_ip, args.clientport)
    print(f"\nWill be comunicating with client on {args.client_ip}:{args.clientport}")
    
    # client = udp_client.SimpleUDPClient("127.0.0.1", args.clientport)
    
    client.send_message("/ready", True)

    # Start the server
    start_server(args.server_ip, args.serverport)
