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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
        loss = self.model(waveforms, features = class_indexes, channels_list=channels_list, embedding = embedding)
        self.log("valid_loss", loss)
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
            shuffle=False,
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
        sampling_steps: List[int],
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

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

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
            

def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target
    noise = target_scaled - preds
    s_target = torch.sum(target_scaled**2, dim=-1) + eps
    s_error = torch.sum(noise**2, dim=-1) + eps
    return 10 * torch.log10(s_target / s_error)



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
        
        torch_si_snr = ScaleInvariantSignalNoiseRatio()
        torch_si_sdr = ScaleInvariantSignalDistortionRatio()

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
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
        
        
        # Log the first item of the batch
        for idx in range(self.num_items):
            # Prepare the logging data
            logging_data = {}
            
            mixture_audio = channels_list[0][idx, 0, :].detach().cpu().numpy()[..., np.newaxis]

            # Prepare the original mixture log
            logging_data[f"Mixture_audio"] = wandb.Audio(
                mixture_audio, 
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
            wandb_logger.log(logging_data, step = trainer.global_step+idx)
            
        if is_train:
            pl_module.train()
