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
from ctm.utils import EMAAndScales_Initialiser, create_model_and_diffusion
from ctm.enc_dec_lib import load_feature_extractor
from ctm.sample_util import karras_sample
import torch.nn as nn
from tqdm import tqdm


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
    
class Diffusion(pl.LightningModule):
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
                                                        total_steps=self.cfg.training.max_steps,
                                                        distill_steps_per_iter=self.cfg.diffusion.distill_steps_per_iter,
                                                        ).get_ema_and_scales

            # Load Feature Extractor
            feature_extractor = load_feature_extractor(self.cfg.diffusion, eval=True)

            # Load main model
            self.net, self.diffusion = create_model_and_diffusion(self.cfg, feature_extractor)

            if self.cfg.diffusion.teacher_model_path is not None and not self.cfg.diffusion.self_learn:
                print(f"loading the teacher model from {self.cfg.diffusion.teacher_model_path}")
                self.teacher_model, _ = create_model_and_diffusion(self.cfg, teacher=True)
                if not self.cfg.diffusion.edm_nn_ncsn and not self.cfg.diffusion.edm_nn_ddpm:
                    self.teacher_model.load_state_dict(torch.load(self.cfg.diffusion.teacher_model_path))
                self.teacher_model.eval()
                self.copy_teacher_params_to_model(self.cfg.diffusion)
                self.teacher_model.requires_grad_(False)
                if self.cfg.diffusion.edm_nn_ncsn:
                    self.net.model.map_noise.freqs = self.teacher_model.model.model.map_noise.freqs
                if self.cfg.diffusion.use_fp16:
                    self.teacher_model.convert_to_fp16()
            else:
                self.teacher_model = None

            self.target_model, _ = create_model_and_diffusion(self.cfg)
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

            # for dst, src in zip(self.target_model.parameters(), self.net.parameters()):
            #     dst.data.copy_(src.data)

            if self.cfg.diffusion.edm_nn_ncsn:
                self.target_model.model.map_noise.freqs = self.teacher_model.model.model.map_noise.freqs

            # Initialize EMA models
            self.ema_models = nn.ModuleList()
            for ema_rate in self.cfg.diffusion.ema_rate:
                ema_model, _ = create_model_and_diffusion(self.cfg)
                for param in ema_model.parameters():
                    param.requires_grad = False
                ema_model.load_state_dict(copy.deepcopy(self.net.state_dict()))
                ema_model.eval()
                self.ema_models.append(ema_model)
            self.ema_models.eval()

            self.diffusion.teacher_model = self.teacher_model

        else:
            raise ValueError(f'Preconditioning {cfg.diffusion.preconditioning} does not exist')


        # FID and Inception Score instances are not registered as modules
        self.evaluation_attrs = {}
        for step in self.cfg.diffusion.denoise_steps_to_log:            
            if self.cfg.testing.calc_inception:
                self.evaluation_attrs[f'inception_student_{step}'] = InceptionScore().eval()
                for i in self.cfg.diffusion.ema_rate:
                    self.evaluation_attrs[f"inception_ema_{i}_{step}"] = InceptionScore().eval()
                self.evaluation_attrs[f'inception_target_{step}'] = InceptionScore().eval()

            if self.cfg.testing.calc_fid:
                self.evaluation_attrs[f'fid_student_{step}'] = FrechetInceptionDistance(feature=2048).eval()
                for i in self.cfg.diffusion.ema_rate:
                    self.evaluation_attrs[f"fid_ema_{i}_{step}"] = FrechetInceptionDistance(feature=2048)
                self.evaluation_attrs[f'fid_target_{step}'] = FrechetInceptionDistance(feature=2048).eval()


    def move_evaluation_attrs_to_device(self):
        # Move all eval attributes to the correct device
        for key, value in self.evaluation_attrs.items():
            self.evaluation_attrs[key] = value.to(self.device)

    def on_train_start(self):
        self.move_evaluation_attrs_to_device()

    def on_validation_start(self):
        self.move_evaluation_attrs_to_device()


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

    def calculate_loss(self, images, cond, split="train"):

        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.cfg.diffusion.num_heun_step)]
        diffusion_training_ = [np.random.rand() < self.cfg.diffusion.diffusion_training_frequency]

        model_kwargs = {}
        if self.cfg.data.class_cond:
            model_kwargs["y"] = cond

        if split == "val":
            apply_adaptive_weight_original_value = self.diffusion.args.apply_adaptive_weight
            self.diffusion.args.apply_adaptive_weight = False
            
        losses = self.diffusion.ctm_losses(
            step=self.global_step,
            model=self.net,
            x_start=images,
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
            self.log(f"{split}/{key} mean", values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            self.log(f"{split}/{key} std", values.std().item())

    def training_step(self, batch, _):
        images, cond = batch

        loss = self.calculate_loss(images, cond, split = "train")

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


    @torch.no_grad()
    def sampling(self, model, sampler = 'exact', ctm=None, teacher=False, step=-1, num_samples=-1, batch_size=-1, resize=False, generator=None, class_idx = None):
        if not teacher:
            model.eval()
        if step == -1:
            step = 1
        if batch_size == -1:
            batch_size = self.args.sampling_batch
        if batch_size>num_samples:
            batch_size = num_samples

        all_images = []
        number = 0

        while num_samples > number:
            model_kwargs = {}
            if self.cfg.data.class_cond:
                if class_idx == None:                
                    model_kwargs["y"] = torch.randint(0, self.cfg.data.num_classes, size=(batch_size, ), device=self.device)
                else:
                    model_kwargs["y"] = torch.full((batch_size,), class_idx, device=self.device)

            # print(f"{number} number samples complete")
            if generator != None:
                x_T = generator.randn(*(batch_size, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution),
                            device=self.device) * self.cfg.diffusion.sigma_max
            else:
                x_T = None

            sample = karras_sample(
                diffusion=self.diffusion,
                model=model,
                shape=(batch_size, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution),
                steps=step,
                model_kwargs=model_kwargs, # in case of classes class goes here???
                device=self.device,
                clip_denoised=True if teacher else self.cfg.diffusion.clip_denoised,
                sampler=sampler,
                generator=None,
                teacher=teacher,
                ctm= ctm,
                x_T=x_T if generator != None else self.x_T.to(self.device) if num_samples == -1 else None,
                clip_output=self.cfg.diffusion.clip_output,
                sigma_min=self.cfg.diffusion.sigma_min,
                sigma_max=self.cfg.diffusion.sigma_max,
                train=False,
            )
            if resize:
                sample = torch.nn.functional.interpolate(sample, size=224, mode="bilinear")

            gathered_samples = sample.contiguous()
            all_images += [sample.cpu() for sample in gathered_samples]
            
            number += int(gathered_samples.shape[0])
        if not teacher:
            model.train()

        arr = torch.stack(all_images, axis=0)

        return arr

    def generate_model_output(self, model, sampler, steps, prefix):

        images_to_log = []
        captions = []

        # generate class conditinal logs if prompted per class
        if self.cfg.data.class_cond and self.cfg.diffusion.log_classes:
            for class_idx in range(self.cfg.data.num_classes):
                for step in steps:
                    xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "Teacher" else False, step=step, num_samples=self.cfg.testing.samples, batch_size=self.cfg.testing.batch_size, ctm=True, class_idx = class_idx)
                    xh = (xh * 0.5 + 0.5).clamp(0, 1)

                    class_name = self.cfg.class_names[class_idx]
                    caption = f'{class_name}/{prefix} {step} Steps'

                    images_to_log.append(make_grid(xh, nrow=8).permute(1, 2, 0).cpu().numpy())
                    captions.append(caption)

        # generate random grid class conditional or unconditional
        for step in steps:
            xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "Teacher" else False, step=step, num_samples=self.cfg.testing.samples, batch_size=self.cfg.testing.batch_size, ctm=True, class_idx = None)
            xh = (xh * 0.5 + 0.5).clamp(0, 1)

            caption = f"{prefix} {step} Steps"

            images_to_log.append(make_grid(xh, nrow=8).permute(1, 2, 0).cpu().numpy())
            captions.append(caption)

        return images_to_log, captions

    def validation_step(self, batch, batch_idx):
        images, cond = batch

        loss = self.calculate_loss(images, cond, split = "val")
        self.log("val_loss", loss, sync_dist=True)

        # Log one sample of the original and generated images for the first batch in the epoch
        if batch_idx == 0 and self.global_rank == 0:

            name = self.cfg.data.name
            images_to_log = []
            captions = []

            # Log teacher model denoises with heun and 18 steps (if we need to see)
            if self.cfg.diffusion.check_ctm_denoising_ability:
                new_images_to_log, new_captions = self.generate_model_output(self.net, 'heun', [18], "Teacher")
                images_to_log.extend(new_images_to_log)
                captions.extend(new_captions)


            # Log student model outputs
            new_images_to_log, new_captions = self.generate_model_output(self.net, 'exact',  self.cfg.diffusion.denoise_steps_to_log, "Student")
            images_to_log.extend(new_images_to_log)
            captions.extend(new_captions)

            # Log target model outputs
            new_images_to_log, new_captions = self.generate_model_output(self.target_model, 'exact',  self.cfg.diffusion.denoise_steps_to_log, "Target")
            images_to_log.extend(new_images_to_log)
            captions.extend(new_captions)


            # Log EMA models outputs
            for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
                new_images_to_log, new_captions = self.generate_model_output(ema_model, 'exact',  self.cfg.diffusion.denoise_steps_to_log, f"ema_{ema_rate}")
                images_to_log.extend(new_images_to_log)
                captions.extend(new_captions)

            # log original data
            original_image = (images[:self.cfg.testing.samples] * 0.5 + 0.5).clamp(0, 1)
            original_grid = make_grid(original_image, nrow=8)
            images_to_log.append(original_grid.permute(1, 2, 0).cpu().numpy())
            captions.append("Original")
            ##################################

            self.logger.log_image(f"val_samples_{name}", images_to_log, caption=captions)


        ####################### Logging FID and Iscore ##################################

        x_batch = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        # Update metric for student model
        self.update_metrics(self.net, self.cfg.diffusion.denoise_steps_to_log, 'student', x_batch)

        # Update metric for target model
        self.update_metrics(self.target_model, self.cfg.diffusion.denoise_steps_to_log, 'target', x_batch)

        # Update metric for ema models
        for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
            self.update_metrics(ema_model, self.cfg.diffusion.denoise_steps_to_log, f'ema_{ema_rate}', x_batch)


    def update_metrics(self, model, steps, prefix, x_batch):
        for step in steps:
            xh = self.sampling(model=model, step=step, num_samples=self.cfg.testing.batch_size, batch_size=self.cfg.testing.batch_size, ctm=True)
            xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(self.device)

            if self.cfg.testing.calc_inception:
                self.evaluation_attrs[f'inception_{prefix}_{step}'].update(xh)

            if self.cfg.testing.calc_fid:
                self.evaluation_attrs[f'fid_{prefix}_{step}'].update(x_batch, real=True)
                self.evaluation_attrs[f'fid_{prefix}_{step}'].update(xh, real=False)

    def on_validation_epoch_end(self):

        # Log Inception scores
        if self.cfg.testing.calc_inception:
            for step in self.cfg.diffusion.denoise_steps_to_log:
                iscore = self.evaluation_attrs[f'inception_student_{step}'].compute()[0]
                self.log(f'iscore/student_{step}', iscore, sync_dist=True)
                self.evaluation_attrs[f'inception_student_{step}'].reset()
                
                for i in self.cfg.diffusion.ema_rate:
                    iscore_ema = self.evaluation_attrs[f"inception_ema_{i}_{step}"].compute()[0]
                    self.log(f'iscore/ema_{i}_{step}', iscore_ema, sync_dist=True)
                    self.evaluation_attrs[f"inception_ema_{i}_{step}"].reset()
                
                iscore_target = self.evaluation_attrs[f'inception_target_{step}'].compute()[0]
                self.log(f'iscore/target_{step}', iscore_target, sync_dist=True)
                self.evaluation_attrs[f'inception_target_{step}'].reset()

        # Log FID scores
        if self.cfg.testing.calc_fid:
            for step in self.cfg.diffusion.denoise_steps_to_log:
                fid = self.evaluation_attrs[f'fid_student_{step}'].compute().item()
                self.log(f'fid/student_{step}', fid, sync_dist=True)
                self.evaluation_attrs[f'fid_student_{step}'].reset()
                
                for i in self.cfg.diffusion.ema_rate:
                    fid_ema = self.evaluation_attrs[f"fid_ema_{i}_{step}"].compute().item()
                    self.log(f'fid/ema_{i}_{step}', fid_ema, sync_dist=True)
                    self.evaluation_attrs[f"fid_ema_{i}_{step}"].reset()
                
                fid_target = self.evaluation_attrs[f'fid_target_{step}'].compute().item()
                self.log(f'fid/target_{step}', fid_target, sync_dist=True)
                self.evaluation_attrs[f'fid_target_{step}'].reset()

        return super().on_validation_epoch_end()

    
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

        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.cfg.training.warmup_epochs, max_iters=self.cfg.training.max_epochs)

        return {
        "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }
    
    