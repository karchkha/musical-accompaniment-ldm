from typing import *
import numpy as np

import pytorch_lightning as pl
import torch
import wandb
from audio_diffusion_pytorch_ import AudioDiffusionModel, AudioDiffusionConditional, Sampler, Schedule
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger #LoggerCollection, WandbLogger
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
import os
import torchaudio
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import shutil
from audioldm_eval import EvaluationHelper
from pathlib import Path

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = AudioDiffusionModel(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("valid_loss", loss)
        return loss



class Audio_DM_Model(pl.LightningModule):
    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, class_cond: bool, separation: bool = False, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.class_cond = class_cond
        self.separation = separation

        # if self.class_cond:
        #     self.model = AudioDiffusionConditional(*args, **kwargs)
        # else:
        self.model = AudioDiffusionModel(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return optimizer

    def get_input(self, batch):

        if isinstance(batch, (list, tuple)) and self.class_cond and self.separation:
            waveforms, class_indexes, stems  = batch

            batch_size, channels, feature_width = waveforms.shape
            mixture = stems.sum(1)

            # Desired output sizes for each layer
            target_sizes = [262144, 4096, 1024, 256, 128, 64, 32]  # TODO: this needs to be caclulated automaticaly somehow

            # Create downscaled versions of waveforms using interpolation
            channels_list = []
            for size in target_sizes:
                if feature_width == size:
                    # No need to resize if the current size matches the target
                    channels_list.append(mixture)
                else:
                    # Resize waveform to the target size
                    resized_mixture = F.interpolate(mixture, size=(size,), mode='linear', align_corners=False)
                    channels_list.append(resized_mixture)
                    # feature_width = size  # Update current length for the next iteration


            # Adjust channel dimensions to match `context_channels`
            # This involves expanding the channel dimension after downsampling
            channels_list = [
                torch.cat([channels_list[i]] * num, dim=1) if num != channels_list[i].shape[1]
                else channels_list[i]
                for i, num in enumerate([1, 512, 1024, 1024, 1024, 1024, 1024])
            ]

            # embedding = torch.randn(2, 4, 32).to(self.device)
            embedding = None

            
        elif isinstance(batch, (list, tuple)) and self.class_cond:
            waveforms, class_indexes, _ = batch
            channels_list = None
            embedding = None
        elif isinstance(batch, (list, tuple)) :
            waveforms, _, _= batch
            class_indexes = None
            channels_list = None  
            embedding = None          
        else:
            waveforms = batch
            class_indexes = None
            channels_list = None
            embedding = None
        return waveforms, class_indexes, channels_list, embedding

    def training_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
        loss = self.model(waveforms, features = class_indexes, channels_list=channels_list, embedding = embedding)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
        loss = self.model(waveforms, features = class_indexes, channels_list=channels_list, embedding = embedding)
        self.log("valid_loss", loss, sync_dist=True)
        return loss




""" Datamodule """

class DatamoduleWithValidation(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # if isinstance(trainer.logger, LoggerCollection):
    #     for logger in trainer.logger:
    #         if isinstance(logger, WandbLogger):
    #             return logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: Union[List[int], int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
        self.diffusion_schedule = diffusion_schedule
        self.diffusion_sampler = diffusion_sampler

        self.log_next = False
        
        
    #     # Check if diffusion_sampler is an instance of KarrasDenoiser
    #     if isinstance(self.diffusion_sampler, KarrasDenoiser):
    #         print("diffusion_sampler is an instance of KarrasDenoiser")
    #         schedule_sampler = create_named_schedule_sampler(self.diffusion_sampler.args, self.diffusion_sampler.args.schedule_sampler, self.diffusion_sampler.args.start_scales)
    #         diffusion_schedule_sampler = create_named_schedule_sampler(self.diffusion_sampler.args, self.diffusion_sampler.args.diffusion_schedule_sampler, self.diffusion_sampler.args.start_scales)
            
    #         self.diffusion_sampler.schedule_sampler = schedule_sampler
    #         self.diffusion_sampler.diffusion_schedule_sampler = diffusion_schedule_sampler
    #         self.scenario = "KarrasDenoiser"
    #     else:
    #         print("diffusion_sampler is NOT an instance of KarrasDenoiser")
    #         self.scenario = "ADPM2Sampler"

    # def sample_wrapper(self, model, noise, step, model_kwargs={}, sampler="heun"):
    #     if self.scenario == 'KarrasDenoiser':
    #         sample = karras_sample(
    #             diffusion=self.diffusion_sampler,
    #             model=model,
    #             shape=(noise.shape[0], self.channels, self.length),
    #             steps=step,
    #             model_kwargs=model_kwargs,  # in case of classes class goes here
    #             device=noise.device,
    #             clip_denoised=True,
    #             sampler=sampler,
    #             generator=None,
    #             teacher=False,
    #             ctm=True,
    #             x_T=noise,
    #             clip_output=True,
    #             sigma_min=self.diffusion_sampler.args.sigma_min,
    #             sigma_max=self.diffusion_sampler.args.sigma_max,
    #             train=False,
    #         )
    #     elif self.scenario == 'ADPM2Sampler':
    #         sample = model.sample(
    #             noise=noise,
    #             sampler=self.diffusion_sampler,
    #             sigma_schedule=self.diffusion_schedule,
    #             num_steps=step,
    #         )
    #     else:
    #         raise ValueError(f"Unknown scenario: {self.scenario}")
    #     return sample

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False
            
        self.save_sample(trainer, pl_module, batch, batch_idx)

    def save_sample(self, trainer, pl_module, batch, batch_idx):
        current_epoch = trainer.current_epoch
        
        new_sampling_rate = 16000 # because FAD is calculated of 16000
        
        # Create base directory path
        base_dir = os.path.dirname(pl_module._trainer.checkpoint_callback.dirpath)
        resampler = torchaudio.transforms.Resample(self.sampling_rate, new_sampling_rate)
        
        # doing this for sweep to work
        if type(self.sampling_steps) == int:
            sampling_steps = [self.sampling_steps]
        else:
            sampling_steps = self.sampling_steps


        # Generate model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.diffusion_sampler, sampling_steps, "net", batch)
        
        for step_idx, step in enumerate(sampling_steps):
            step_dir = os.path.join(base_dir, f'audios_{current_epoch}_step{step}')
            generated_dir = os.path.join(step_dir, 'generated')
            original_dir = os.path.join(step_dir, 'original')

            # Create directories if they don't exist
            os.makedirs(generated_dir, exist_ok=True)
            os.makedirs(original_dir, exist_ok=True)
            
            for idx in range(batch[0].size(0)):
                audio = new_audios_to_log[step_idx][idx]  # Ensure tensor is on CPU
                original_audio = batch[0][idx].cpu()  # Ensure tensor is on CPU

                # Resample audio
                resampled_audio = resampler(torch.tensor(audio).permute(1,0))
                resampled_original_audio = resampler(original_audio.detach())

                # Define file names
                generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')

                # Save audio files
                torchaudio.save(generated_file_name, resampled_audio, 16000)
                torchaudio.save(original_file_name, resampled_original_audio, 16000)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        base_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
        wandb_logger = get_wandb_logger(trainer).experiment
        evaluator = EvaluationHelper(sampling_rate=16000, device=pl_module.device)

        sampling_steps = self.sampling_steps if isinstance(self.sampling_steps, list) else [self.sampling_steps]

        for step in sampling_steps:
            step_dir = os.path.join(base_dir, f'audios_{current_epoch}_step{step}')
            if os.path.exists(step_dir):
                dir1, dir2 = Path(os.path.join(step_dir, "generated")), Path(os.path.join(step_dir, "original"))
                print("\nNow evaluating:", step_dir)
                metrics = evaluator.main(str(dir1), str(dir2))
                metrics_buffer = {f"step_{step}/{k}" if isinstance(self.sampling_steps, list) else k: float(v) for k, v in metrics.items()}

                if metrics_buffer:
                    for k, v in metrics_buffer.items():
                        wandb_logger.log({k: v}, commit=False)
                        print(k, v)
                    wandb_logger.log({}, commit=True)
                shutil.rmtree(dir1)
                shutil.rmtree(dir2)      
    
    def generate_model_output(self, model, sampler, steps, prefix, batch):

        audios_to_log = []
        captions = []
        
        batch_size = batch[0].size(0)

        # generate random grid class conditional or unconditional
        for step in steps:
            xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "teacher_model" else False, prefix=prefix, step=step, num_samples=1, batch_size=batch_size, ctm= False, class_idx = None)
            xh.clamp(-1.0, 1.0) # (xh * 0.5 + 0.5).clamp(0, 1)

            caption = f"{prefix}_{step}_Steps"

            audios_to_log.append(xh.permute(0, 2, 1).cpu().numpy())
            captions.append(caption)

        return audios_to_log, captions

    @torch.no_grad()
    def sampling(self, model, sampler = 'exact', ctm=None, teacher=False, prefix="", step=-1, num_samples=-1, batch_size=-1, resize=False, generator=None, class_idx = None):
        # if not teacher:
        #     model.eval()
        if step == -1:
            step = 1
        if batch_size == -1:
            batch_size = model.cfg.datamodule.batch_size

        all_images = []
        number = 0
        
        # Dynamically select model based on prefix using getattr
        model_to_use = model.model

        while num_samples > number:
            model_kwargs = {}
            is_train = model.training
            if is_train:
                model.eval()

            # Get start diffusion noise
            noise = torch.randn(
                (batch_size, self.channels, self.length), device=model.device
            )

            # samples = self.sample_wrapper(model, noise, step)

            samples = model_to_use.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=step,
            )
            sample = samples #rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            if is_train:
                model.train()

            gathered_samples = sample.contiguous()
            all_images += [sample.cpu() for sample in gathered_samples]
            
            number += int(gathered_samples.shape[0])
        # if not teacher:
        #     model.train()

        arr = torch.stack(all_images, axis=0)

        return arr


    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )
        
        # doing this for sweep to work
        if type(self.sampling_steps) == int:
            sampling_steps = [self.sampling_steps]
        else:
            sampling_steps = self.sampling_steps

        for steps in sampling_steps:
            samples = model.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )
            samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            wandb_logger.log(
                {
                    f"sample_{idx}_{steps}": wandb.Audio(
                        samples[idx],
                        caption=f"Sampled in {steps} steps",
                        sample_rate=self.sampling_rate,
                    )
                    for idx in range(self.num_items)
                }
            )

        if is_train:
            pl_module.train()


class MultiSourceSampleLogger(SampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        stems: List[str]
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            diffusion_sampler=diffusion_sampler,
        )
        self.stems = stems

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )

        for steps in self.sampling_steps:
            samples = model.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )
            samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            for i in range(samples.shape[-1]):
                wandb_logger.log(
                 {
                     f"sample_{self.stems[i]}_{idx}_{steps}": wandb.Audio(
                            samples[idx, :, i][..., np.newaxis],
                            caption=f"Sampled in {steps} steps",
                            sample_rate=self.sampling_rate,
                     ) for idx in range(self.num_items)
                    }
            )
            # log mixture
            wandb_logger.log(
                {
                    f"sample_mix_{idx}_{steps}": wandb.Audio(
                        samples[idx, :, :].sum(axis=-1, keepdims=True),
                        caption=f"Sampled in {steps} steps",
                        sample_rate=self.sampling_rate,
                    ) for idx in range(self.num_items)
                })
        if is_train:
            pl_module.train()



class ClassCondTrackSampleLogger(SampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        stems: List[str]
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            diffusion_sampler=diffusion_sampler,
        )
        self.stems = stems

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )

        # Iterate over each diffusion step size
        for steps in self.sampling_steps:
            # Iterate over each one-hot encoded feature vector (each stem)
            for i, stem in enumerate(self.stems):
                # Create a feature tensor for the current stem for all items
                current_features = torch.zeros(self.num_items, len(self.stems)).to(pl_module.device)
                current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)

                # Sample from the model using the noise and the current one-hot features
                samples = model.sample(
                    noise=noise,
                    features=current_features,
                    sampler=self.diffusion_sampler,
                    sigma_schedule=self.diffusion_schedule,
                    num_steps=steps,
                )
                samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

                # Log each sample for the current stem
                for idx in range(self.num_items):
                    audio_data = samples[idx, :, 0][..., np.newaxis]  # Reshape for mono audio
                    wandb_logger.log({
                        f"sample_{stem}_{idx}_{steps}": wandb.Audio(
                            audio_data,
                            caption=f"Sampled in {steps} steps",
                            sample_rate=self.sampling_rate
                        )
                    })



        # for steps in self.sampling_steps:
        #     samples = model.sample(
        #         noise=noise,
        #         sampler=self.diffusion_sampler,
        #         sigma_schedule=self.diffusion_schedule,
        #         num_steps=steps,
        #     )
        #     samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            # for i in range(samples.shape[-1]):
            #     wandb_logger.log(
            #      {
            #          f"sample_{self.stems[i]}_{idx}_{steps}": wandb.Audio(
            #                 samples[idx, :, i][..., np.newaxis],
            #                 caption=f"Sampled in {steps} steps",
            #                 sample_rate=self.sampling_rate,
            #          ) for idx in range(self.num_items)
            #         }
            # )
            # # log mixture
        #     wandb_logger.log(
        #         {
        #             f"sample_mix_{idx}_{steps}": wandb.Audio(
        #                 samples[idx, :, :].sum(axis=-1, keepdims=True),
        #                 caption=f"Sampled in {steps} steps",
        #                 sample_rate=self.sampling_rate,
        #             ) for idx in range(self.num_items)
        #         })
        if is_train:
            pl_module.train()
            



class ClassCondSeparateTrackSampleLogger(SampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        stems: List[str]
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            diffusion_sampler=diffusion_sampler,
        )
        self.stems = stems
        
        self.torch_si_snr = ScaleInvariantSignalNoiseRatio()
        self.torch_si_sdr = ScaleInvariantSignalDistortionRatio()
        
        # Initialize metrics dictionaries for each number of steps and each stem
        self.metrics_log = {
            steps: {stem: {'si_snr': [], 'si_sdr': [], 'msdm_si_snr': []} for stem in stems} for steps in sampling_steps
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        # if self.log_next:
        #     self.log_sample(trainer, pl_module, batch)
        #     self.log_next = False
        
        # if self.log_next == False and batch_idx % 5 == 0:
        #     self.update_metrics(trainer, pl_module, batch)

        if batch_idx % 5 == 0:
            self.log_sample(trainer, pl_module, batch, batch_idx)

    def log_sample(self, trainer, pl_module, batch, batch_idx):
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        original_samples, generated_samples, mixture_audios = self.generate_sample(trainer, pl_module, batch)
        
        if  batch_idx ==0 and trainer.is_global_zero:
            self.log_audio(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)
        
        # if trainer.is_global_zero: # TODO this need to be changed
        self.update_metrics(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)       
        
        if is_train:
            pl_module.train()

    def generate_sample(self, trainer, pl_module, batch):
        
        model = pl_module.model
        
        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding = pl_module.get_input(batch)

        # Get start diffusion noise for whole batch
        noise = torch.randn(
            (waveforms.size(0), self.channels, self.length), device=pl_module.device
        )

        # Dictionary to store generated samples
        generated_samples = {stem: {} for stem in self.stems}
        
        
        # Iterate over each diffusion step size
        for steps in self.sampling_steps:
            # Iterate over each one-hot encoded feature vector (each stem)
            for i, stem in enumerate(self.stems):
                # Create a feature tensor for the current stem for all items
                current_features = torch.zeros(waveforms.size(0), len(self.stems)).to(pl_module.device)
                current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)

                # Sample from the model using the noise and the current one-hot features
                samples = model.sample(
                    noise=noise,
                    features=current_features,
                    sampler=self.diffusion_sampler,
                    sigma_schedule=self.diffusion_schedule,
                    num_steps=steps,
                    channels_list=channels_list
                )
                samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

                # Store the generated samples
                for idx in range(waveforms.size(0)):
                    if steps not in generated_samples[stem]:
                        generated_samples[stem][steps] = []
                    generated_samples[stem][steps].append(samples[idx])

        # get original stems
        original_samples = {stem: {} for stem in self.stems}
        
        original_stems = batch[2]
        
        for i, stem in enumerate(self.stems):
            stem_data = original_stems[:, i]
            stem_data =rearrange(stem_data, "b c t -> b t c") .detach().cpu().numpy()
            original_samples[stem] = []
            for idx in range(waveforms.size(0)):
                original_samples[stem].append(stem_data[idx])  
        
        mixture_audios = batch[2].sum(1)[:, 0, :].detach().cpu().numpy()[..., np.newaxis] #channels_list[0][idx, 0, :].detach().cpu().numpy()[..., np.newaxis]
        
        return  original_samples, generated_samples, mixture_audios


    @torch.no_grad()
    def log_audio(self, original_samples, generated_samples, mixture_audio, wandb_logger, trainer):
        
        # Log the first item of the batch
        for idx in range(self.num_items):
            # Prepare the logging data
            logging_data = {}

            # Prepare the original mixture log
            logging_data[f"Mixture_audio"] = wandb.Audio(
                mixture_audio[idx], 
                caption=f"Mixture Audio {idx}", 
                sample_rate=self.sampling_rate
            )

            # Prepare the original stems logs
            for stem in self.stems:
                original_audio = original_samples[stem][idx]
                logging_data[f"original_{stem}"] = wandb.Audio(
                    original_audio,
                    caption=f"Original {stem} Audio {idx}",
                    sample_rate=self.sampling_rate
                )

            # Prepare each generated sample for the current stem by number of steps
            for steps in self.sampling_steps:
                for stem in self.stems:
                    generated_audio = generated_samples[stem][steps][idx]
                    logging_data[f"generated_{stem}_steps_{steps}"] = wandb.Audio(
                        generated_audio,
                        caption=f"{stem} Sampled in {steps} steps (idx: {idx})",
                        sample_rate=self.sampling_rate
                    )

                # Prepare the mixture of the generated samples
                mix_audio = sum(generated_samples[stem][steps][idx] for stem in self.stems)
                logging_data[f"generated_mix_steps_{steps}"] = wandb.Audio(
                    mix_audio,
                    caption=f"Sampled in {steps} steps (Mix) (idx: {idx})",
                    sample_rate=self.sampling_rate
                )

            # Log all accumulated data
            wandb_logger.log(logging_data) #step = trainer.global_step+idx)
            
    def sisnr(self, preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
        target_scaled = alpha * target
        noise = target_scaled - preds
        s_target = torch.sum(target_scaled**2, dim=-1) + eps
        s_error = torch.sum(noise**2, dim=-1) + eps
        return 10 * torch.log10(s_target / s_error)


    def sliding_window(self, tensor, window_size=1024, step_size=512):
        num_windows = (tensor.size(-1) - window_size) // step_size + 1
        windows = []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            windows.append(tensor[..., start:end])
        return torch.stack(windows, dim=0)        
            
    @torch.no_grad()
    def update_metrics(self, original_samples, generated_samples, mixture_audios, wandb_logger, trainer):

        for steps in self.sampling_steps:
            for stem in self.stems:
                for idx in range(len(original_samples[stem])):  # Iterate over each sample in the batch
                    original_audio = original_samples[stem][idx]
                    generated_audio = generated_samples[stem][steps][idx]
                    mixture_audio = mixture_audios[idx]

                    original_audio = torch.tensor(original_audio).permute(1, 0)
                    generated_audio = torch.tensor(generated_audio).permute(1, 0)
                    mixture_audio = torch.tensor(mixture_audio).permute(1, 0)

                    si_snr = self.torch_si_snr(generated_audio, original_audio)
                    si_sdr = self.torch_si_sdr(generated_audio, original_audio)
                    msdm_si_snr = self.sisnr(generated_audio, original_audio) - self.sisnr(mixture_audio, original_audio)

                    # # Apply sliding window
                    # original_windows = self.sliding_window(original_audio)
                    # generated_windows = self.sliding_window(generated_audio)
                    # mixture_windows = self.sliding_window(mixture_audio)

                    # msdm_si_snr_values = []
                    # for ow, gw, mw in zip(original_windows, generated_windows, mixture_windows):
                    #     msdm_si_snr = self.sisnr(gw, ow).mean().item() - self.sisnr(mw, ow).mean().item()
                    #     msdm_si_snr_values.append(msdm_si_snr)


                    # mean_msdm_si_snr = 0.0 #self.torch_si_snr(generated_audio, original_audio) - self.torch_si_snr(mixture_audio, original_audio) # sum(msdm_si_snr_values) / len(msdm_si_snr_values)

                    # Append computed metrics to the corresponding lists
                    # self.metrics_log[steps][stem]['msdm_si_snr'].append(mean_msdm_si_snr)



                    # Append computed metrics to the corresponding lists
                    self.metrics_log[steps][stem]['si_snr'].append(si_snr.item())
                    self.metrics_log[steps][stem]['si_sdr'].append(si_sdr.item())
                    self.metrics_log[steps][stem]['msdm_si_snr'].append(msdm_si_snr.item())

    
    def on_validation_epoch_end(self, trainer, pl_module):
        # wandb_logger = get_wandb_logger(trainer).experiment
        log_dict = {}
        
        for steps in self.sampling_steps:
            for stem in self.stems:
                mean_si_snr = sum(self.metrics_log[steps][stem]['si_snr']) / len(self.metrics_log[steps][stem]['si_snr'])
                mean_si_sdr = sum(self.metrics_log[steps][stem]['si_sdr']) / len(self.metrics_log[steps][stem]['si_sdr'])
                mean_msdm_si_snr = sum(self.metrics_log[steps][stem]['msdm_si_snr']) / len(self.metrics_log[steps][stem]['msdm_si_snr'])

                # wandb_logger.log({
                #     f'{stem}_mean_si_snr_{steps}_steps': mean_si_snr,
                #     f'{stem}_mean_si_sdr_{steps}_steps': mean_si_sdr
                # })
                log_dict[f'si_snr/{stem}_{steps}_steps'] = mean_si_snr
                log_dict[f'si_sdr/{stem}_{steps}_steps'] = mean_si_sdr
                log_dict[f'msdm_si_snr/{stem}_{steps}_steps'] = mean_msdm_si_snr


                # Reset metrics for the current number of steps and stem
                self.metrics_log[steps][stem]['si_snr'] = []
                self.metrics_log[steps][stem]['si_sdr'] = []
                self.metrics_log[steps][stem]['msdm_si_snr'] = []
        
        pl_module.log_dict(log_dict, sync_dist=True) # step=trainer.global_step)

