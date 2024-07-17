import pytorch_lightning as pl
# from networks import VEPrecond, VPPrecond, EDMPrecond, CTPrecond, iCTPrecond
# from loss import VELoss, VPLoss, EDMLoss, CTLoss, iCTLoss
from torch import optim
import numpy as np
import torch
# from sampler import multistep_consistency_sampling
from torchvision.utils import make_grid, save_image
import copy
from torchmetrics.image.inception import InceptionScore
# from sampler import multistep_consistency_sampling
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from ctm.utils import EMAAndScales_Initialiser, create_model_and_diffusion_audio
from ctm.enc_dec_lib import load_feature_extractor
from ctm.sample_util import karras_sample
import torch.nn as nn
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback, Trainer
import torch.nn.functional as F
from typing import *
from einops import rearrange


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class Audio_CTM_Model(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        if cfg.diffusion.preconditioning == 'ctm':

            self.ema_scale_fn = EMAAndScales_Initialiser(target_ema_mode=self.cfg.diffusion.target_ema_mode,
                                                        start_ema=self.cfg.diffusion.start_ema,
                                                        scale_mode=self.cfg.diffusion.scale_mode,
                                                        start_scales=self.cfg.diffusion.start_scales,
                                                        end_scales=self.cfg.diffusion.end_scales,
                                                        total_steps=self.cfg.trainer.max_steps,
                                                        distill_steps_per_iter=self.cfg.diffusion.distill_steps_per_iter,
                                                        ).get_ema_and_scales

            # Load Feature Extractor
            feature_extractor = load_feature_extractor(self.cfg.diffusion, eval=True)

            # Load main model
            self.net, self.diffusion = create_model_and_diffusion_audio(self.cfg, feature_extractor)

            if self.cfg.diffusion.teacher_model_path is not None and not self.cfg.diffusion.self_learn:
                print(f"loading the teacher model from {self.cfg.diffusion.teacher_model_path}")
                self.teacher_model, _ = create_model_and_diffusion_audio(self.cfg, teacher=True)
                # if not self.cfg.diffusion.edm_nn_ncsn and not self.cfg.diffusion.edm_nn_ddpm:
                ckpt = torch.load(self.cfg.diffusion.teacher_model_path, map_location="cpu")
                # ckpt = self.adjust_state_dict(ckpt["state_dict"])
                self.teacher_model.load_state_dict(ckpt["state_dict"],strict=False)
                self.teacher_model.eval()
                self.copy_teacher_params_to_model(self.cfg.diffusion)
                self.teacher_model.requires_grad_(False)
                # if self.cfg.diffusion.edm_nn_ncsn:
                #     self.net.model.map_noise.freqs = self.teacher_model.model.model.map_noise.freqs
                if self.cfg.diffusion.use_fp16:
                    self.teacher_model.convert_to_fp16()
            else:
                self.teacher_model = None

            self.target_model, _ = create_model_and_diffusion_audio(self.cfg)
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

            # for dst, src in zip(self.target_model.parameters(), self.net.parameters()):
            #     dst.data.copy_(src.data)

            # if self.cfg.diffusion.edm_nn_ncsn:
            #     self.target_model.model.map_noise.freqs = self.teacher_model.model.model.map_noise.freqs

            # Initialize EMA models
            self.ema_models = nn.ModuleList()
            for ema_rate in self.cfg.diffusion.ema_rate:
                ema_model, _ = create_model_and_diffusion_audio(self.cfg)
                for param in ema_model.parameters():
                    param.requires_grad = False
                ema_model.load_state_dict(copy.deepcopy(self.net.state_dict()))
                ema_model.eval()
                self.ema_models.append(ema_model)
            self.ema_models.eval()

            self.diffusion.teacher_model = self.teacher_model

        else:
            raise ValueError(f'Preconditioning {cfg.diffusion.preconditioning} does not exist')

        # extract class cond and separation arguments from sub-networks
        self.class_cond = self.net.model.class_cond
        self.separation = self.net.model.separation
        # # FID and Inception Score instances are not registered as modules
        # self.evaluation_attrs = {}
        # for step in self.cfg.diffusion.denoise_steps_to_log:            
        #     if self.cfg.testing.calc_inception:
        #         self.evaluation_attrs[f'inception_student_{step}'] = InceptionScore().eval()
        #         for i in self.cfg.diffusion.ema_rate:
        #             self.evaluation_attrs[f"inception_ema_{i}_{step}"] = InceptionScore().eval()
        #         self.evaluation_attrs[f'inception_target_{step}'] = InceptionScore().eval()

        #     if self.cfg.testing.calc_fid:
        #         self.evaluation_attrs[f'fid_student_{step}'] = FrechetInceptionDistance(feature=2048).eval()
        #         for i in self.cfg.diffusion.ema_rate:
        #             self.evaluation_attrs[f"fid_ema_{i}_{step}"] = FrechetInceptionDistance(feature=2048)
        #         self.evaluation_attrs[f'fid_target_{step}'] = FrechetInceptionDistance(feature=2048).eval()


    # def adjust_state_dict(self, state_dict):
    #     # adjust pretrained teacher's format to ours/
    #     new_state_dict = {}
    #     for key in state_dict.keys():
    #         if key.startswith("model.unet."):
    #             new_key = key.replace("model.unet.", "model.")
    #         # else:
    #         #     new_key = key
    #         new_state_dict[new_key] = state_dict[key]
    #     return new_state_dict

    # def move_evaluation_attrs_to_device(self):
    #     # Move all eval attributes to the correct device
    #     for key, value in self.evaluation_attrs.items():
    #         self.evaluation_attrs[key] = value.to(self.device)

    # def on_train_start(self):
    #     self.move_evaluation_attrs_to_device()

    # def on_validation_start(self):
    #     self.move_evaluation_attrs_to_device()


    def copy_teacher_params_to_model(self, args):
        def filter_(dst_name):
            dst_ = dst_name.split('.')
            for idx, name in enumerate(dst_):
                if '_train' in name:
                    dst_[idx] = ''.join(name.split('_train'))
            return '.'.join(dst_)

        for dst_name, dst in self.net.named_parameters():
            for src_name, src in self.teacher_model.named_parameters():
                if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                    dst.data.copy_(src.data)
                    if args.linear_probing:
                        dst.requires_grad = False
                    break
                if args.linear_probing:
                    if filter_(dst_name) in ['.'.join(src_name.split('.')[1:]), src_name]:
                        dst.data.copy_(src.data)
                        break


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


    def calculate_loss(self, waveforms, features, channels_list, embedding, split="train"):

        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.cfg.diffusion.num_heun_step)]
        diffusion_training_ = [np.random.rand() < self.cfg.diffusion.diffusion_training_frequency]

        model_kwargs = {}
        # if self.cfg.model.class_cond:
        model_kwargs["features"] = features
        model_kwargs["channels_list"] = channels_list
        model_kwargs["embedding"] = embedding

        if split == "val":
            apply_adaptive_weight_original_value = self.diffusion.args.apply_adaptive_weight
            self.diffusion.args.apply_adaptive_weight = False
            
        losses = self.diffusion.ctm_losses(
            step=self.global_step,
            model=self.net,
            x_start=waveforms,
            model_kwargs=model_kwargs,
            target_model=self.target_model,
            discriminator=None,
            init_step=0,              # TODO: might need to change to train start stpe (resume_step) if we adopt any schedulers for GAN ir something similar 
            ctm=True,
            num_heun_step=num_heun_step[0],
            gan_num_heun_step=-1,
            diffusion_training_=diffusion_training_[0],
            gan_training_=False
        )

        if split == "val":
            self.diffusion.args.apply_adaptive_weight = apply_adaptive_weight_original_value

        if 'consistency_loss' in list(losses.keys()):
            # print("Consistency learning")
            loss = self.cfg.diffusion.consistency_weight * losses["consistency_loss"].mean()

            if 'denoising_loss' in list(losses.keys()):
                loss = loss + self.cfg.diffusion.denoising_weight * losses['denoising_loss'].mean()

            self.log_loss_dict( {k: v.view(-1) for k, v in losses.items()}, split)

            # self.mp_trainer.backward(loss)
           
        elif 'denoising_loss' in list(losses.keys()):
            loss = losses['denoising_loss'].mean()
            self.log_loss_dict({k: v.view(-1) for k, v in losses.items()}, split)
            # self.mp_trainer.backward(loss)

        return loss
    
    def log_loss_dict(self, losses, split):
        for key, values in losses.items():
            self.log(f"{split}/{key} mean", values.mean().item(), sync_dist=True)
            # Log the quantiles (four quartiles, in particular).
            self.log(f"{split}/{key} std", values.std().item(), sync_dist=True)

    def training_step(self, batch, _):
        waveforms, features, channels_list, embedding = self.get_input(batch)

        loss = self.calculate_loss(waveforms, features, channels_list, embedding, split = "train")


        return loss

    def on_train_batch_end(self, out, batch, batch_idx):
        # Update EMA models manually at the end of each batch

        # Update target model
        target_ema, _ = self.ema_scale_fn(self.global_step)
        self.update_ema(self.target_model, self.net, target_ema)

        # Update all the EMA models
        for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
            self.update_ema(ema_model, self.net, ema_rate)

    def update_ema(self, ema_model, model, decay):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)



    def validation_step(self, batch, batch_idx):
        waveforms, features, channels_list, embedding = self.get_input(batch)
        # images, cond = batch

        loss = self.calculate_loss(waveforms, features, channels_list, embedding, split = "val")
        self.log("val_loss", loss, sync_dist=True)




    # def on_validation_epoch_end(self):

    #     # Log Inception scores
    #     if self.cfg.testing.calc_inception:
    #         for step in self.cfg.diffusion.denoise_steps_to_log:
    #             iscore = self.evaluation_attrs[f'inception_student_{step}'].compute()[0]
    #             self.log(f'iscore/student_{step}', iscore, sync_dist=True)
    #             self.evaluation_attrs[f'inception_student_{step}'].reset()
                
    #             for i in self.cfg.diffusion.ema_rate:
    #                 iscore_ema = self.evaluation_attrs[f"inception_ema_{i}_{step}"].compute()[0]
    #                 self.log(f'iscore/ema_{i}_{step}', iscore_ema, sync_dist=True)
    #                 self.evaluation_attrs[f"inception_ema_{i}_{step}"].reset()
                
    #             iscore_target = self.evaluation_attrs[f'inception_target_{step}'].compute()[0]
    #             self.log(f'iscore/target_{step}', iscore_target, sync_dist=True)
    #             self.evaluation_attrs[f'inception_target_{step}'].reset()

    #     # Log FID scores
    #     if self.cfg.testing.calc_fid:
    #         for step in self.cfg.diffusion.denoise_steps_to_log:
    #             fid = self.evaluation_attrs[f'fid_student_{step}'].compute().item()
    #             self.log(f'fid/student_{step}', fid, sync_dist=True)
    #             self.evaluation_attrs[f'fid_student_{step}'].reset()
                
    #             for i in self.cfg.diffusion.ema_rate:
    #                 fid_ema = self.evaluation_attrs[f"fid_ema_{i}_{step}"].compute().item()
    #                 self.log(f'fid/ema_{i}_{step}', fid_ema, sync_dist=True)
    #                 self.evaluation_attrs[f"fid_ema_{i}_{step}"].reset()
                
    #             fid_target = self.evaluation_attrs[f'fid_target_{step}'].compute().item()
    #             self.log(f'fid/target_{step}', fid_target, sync_dist=True)
    #             self.evaluation_attrs[f'fid_target_{step}'].reset()

    #     return super().on_validation_epoch_end()

    
    def configure_optimizers(self):
        cfg = self.cfg.optim
        if cfg.optimizer == 'radam':
            optimizer = optim.RAdam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), lr=cfg.lr)
        elif cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=cfg.lr)

        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.cfg.optim.warmup_epochs, max_iters=self.cfg.trainer.max_epochs)

        return {
        "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }
    
    
    
    
    
    
    
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


class UncondSampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        check_ctm_denoising_ability: bool = True,
        clip_output: bool = True,
        clip_denoised: bool = False,
        denoise_steps_to_log: List[int] = [1],
        # diffusion_schedule = None,  # Assuming type Schedule
        # diffusion_sampler = None,  # Assuming type Sampler
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.check_ctm_denoising_ability = check_ctm_denoising_ability
        self.clip_output = clip_output
        self.clip_denoised = clip_denoised
        self.denoise_steps_to_log = denoise_steps_to_log
        # self.diffusion_schedule = diffusion_schedule
        # self.diffusion_sampler = diffusion_sampler

        self.log_next = False

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

        if batch_idx == 0:
            self.log_sample(trainer, pl_module, batch, batch_idx)
            
        # self.save_sample(trainer, pl_module, batch, batch_idx)

    # def log_sample(self, trainer, pl_module, batch, batch_idx):
        
    #     is_train = pl_module.training
    #     if is_train:
    #         pl_module.eval()

    #     wandb_logger = get_wandb_logger(trainer).experiment
    #     original_samples, generated_samples, mixture_audios = self.generate_sample(trainer, pl_module, batch)
        
    #     if  batch_idx ==0 and trainer.is_global_zero:
    #         self.log_audio(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)
        
    #     # if trainer.is_global_zero: # TODO this need to be changed
    #     self.update_metrics(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)       
        
    #     if is_train:
    #         pl_module.train()

    def get_nested_attr(self, obj, attr):
        """ Get a nested attribute, supporting indexed access """
        parts = attr.split('[')
        for part in parts:
            if ']' in part:
                key = int(part.rstrip(']'))
                obj = obj[key]
            else:
                obj = getattr(obj, part)
        return obj

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch, batch_idx):
        # is_train = pl_module.training
        # if is_train:
        #     pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        # model = pl_module.model

        # name = self.cfg.data.name
        audios_to_log = []
        captions = []

        # Log teacher model denoises with heun and 18 steps (if we need to see)
        if self.check_ctm_denoising_ability:
            new_audios_to_log, new_captions = self.generate_model_output(pl_module, 'heun', [18], "teacher_model")
            audios_to_log.extend(new_audios_to_log)
            captions.extend(new_captions)

        # Log student model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, 'exact',  self.denoise_steps_to_log, "net")
        audios_to_log.extend(new_audios_to_log)
        captions.extend(new_captions)

        # Log target model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, 'exact',  self.denoise_steps_to_log, "target_model")
        audios_to_log.extend(new_audios_to_log)
        captions.extend(new_captions)


        # Log EMA models outputs
        for i, ema_rate in enumerate(pl_module.cfg.diffusion.ema_rate):
            new_audios_to_log, new_captions = self.generate_model_output(pl_module, 'exact',  self.denoise_steps_to_log, f"ema_models[{i}]")
            audios_to_log.extend(new_audios_to_log)
            captions.extend(new_captions)

        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding = pl_module.get_input(batch)

        # log original data
        original_audio = waveforms[:self.num_items].clamp(-1.0, 1.0)
        audios_to_log.append(original_audio.permute(0, 2, 1).cpu().numpy())
        captions.append("Original")
        ##################################


        # Log one sample of the original and generated images for the first batch in the epoch
        if trainer.is_global_zero and batch_idx==0:            
            for idx in range(self.num_items):
                logging_data = {}
                for i, caption in enumerate(captions):
                    audio = audios_to_log[i][idx]
                    logging_data[caption] = wandb.Audio(audio, sample_rate=self.sampling_rate, caption=f"batch_{batch_idx}_N_{idx}") # TODO: Make validation epoch number here instaed of batch index
                wandb_logger.log(logging_data)
        
        #### TODO: Here will be saving files in seperate folders and then later calculate FAD and other metrics on them!!! 
        
        # @torch.no_grad()
        # def save_sample(self, trainer, pl_module, batch, batch_idx):
            

        #     ####################### Logging FID and Iscore ##################################

        #     x_batch = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        #     # Update metric for student model
        #     self.update_metrics(self.net, self.cfg.diffusion.denoise_steps_to_log, 'student', x_batch)

        #     # Update metric for target model
        #     self.update_metrics(self.target_model, self.cfg.diffusion.denoise_steps_to_log, 'target', x_batch)

        #     # Update metric for ema models
        #     for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
        #         self.update_metrics(ema_model, self.cfg.diffusion.denoise_steps_to_log, f'ema_{ema_rate}', x_batch)


    def update_metrics(self, model, steps, prefix, x_batch):
        for step in steps:
            xh = self.sampling(model=model, step=step, num_samples=self.cfg.testing.batch_size, batch_size=self.cfg.testing.batch_size, ctm=True)
            xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(self.device)

            if self.cfg.testing.calc_inception:
                self.evaluation_attrs[f'inception_{prefix}_{step}'].update(xh)

            if self.cfg.testing.calc_fid:
                self.evaluation_attrs[f'fid_{prefix}_{step}'].update(x_batch, real=True)
                self.evaluation_attrs[f'fid_{prefix}_{step}'].update(xh, real=False)



        # # Get start diffusion noise
        # noise = torch.randn(
        #     (self.num_items, self.channels, self.length), device=pl_module.device
        # )

        # for steps in self.sampling_steps:

        #     samples = model.sample(
        #         noise=noise,
        #         sampler=self.diffusion_sampler,
        #         sigma_schedule=self.diffusion_schedule,
        #         num_steps=steps,
        #     )
        #     samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

        #     wandb_logger.log(
        #         {
        #             f"sample_{idx}_{steps}": wandb.Audio(
        #                 samples[idx],
        #                 caption=f"Sampled in {steps} steps",
        #                 sample_rate=self.sampling_rate,
        #             )
        #             for idx in range(self.num_items)
        #         }
        #     )

        # if is_train:
        #     pl_module.train()
            
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
        model_to_use = self.get_nested_attr(model, prefix)

        while num_samples > number:
            model_kwargs = {}
            # if self.cfg.data.class_cond:
            #     if class_idx == None:                
            #         model_kwargs["y"] = torch.randint(0, self.cfg.data.num_classes, size=(batch_size, ), device=self.device)
            #     else:
            #         model_kwargs["y"] = torch.full((batch_size,), class_idx, device=self.device)

            # print(f"{number} number samples complete")
            if generator != None:
                x_T = generator.randn(*(batch_size, self.channels, self.length),
                            device=self.device) * model.cfg.diffusion.sigma_max
            else:
                x_T = None

            sample = karras_sample(
                diffusion=model.diffusion,
                model=model_to_use,
                shape=(batch_size, self.channels, self.length),
                steps=step,
                model_kwargs=model_kwargs, # in case of classes class goes here???
                device=model.device,
                clip_denoised=True if teacher else self.clip_denoised,
                sampler=sampler,
                generator=None,
                teacher=teacher,
                ctm= ctm,
                x_T=x_T if generator != None else self.x_T.to(self.device) if num_samples == -1 else None,
                clip_output=self.clip_output,
                sigma_min=model.cfg.diffusion.sigma_min,
                sigma_max=model.cfg.diffusion.sigma_max,
                train=False,
            )

            gathered_samples = sample.contiguous()
            all_images += [sample.cpu() for sample in gathered_samples]
            
            number += int(gathered_samples.shape[0])
        # if not teacher:
        #     model.train()

        arr = torch.stack(all_images, axis=0)

        return arr

    def generate_model_output(self, model, sampler, steps, prefix):

        audios_to_log = []
        captions = []

        # generate random grid class conditional or unconditional
        for step in steps:
            xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "teacher_model" else False, prefix=prefix, step=step, num_samples=1, batch_size=model.cfg.datamodule.batch_size, ctm=True, class_idx = None)
            xh.clamp(-1.0, 1.0) # (xh * 0.5 + 0.5).clamp(0, 1)

            caption = f"{prefix} {step} Steps"

            audios_to_log.append(xh.permute(0, 2, 1).cpu().numpy())
            captions.append(caption)

        return audios_to_log, captions
