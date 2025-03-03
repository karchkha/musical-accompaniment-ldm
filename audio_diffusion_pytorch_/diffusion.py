from math import sqrt
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor
import tqdm
import random

from .utils import default, exists
import torchaudio.transforms as T

""" Distributions """


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


""" Schedules """


class Schedule(nn.Module):
    """Interface used by different schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        
        if num_steps == 1:
            sigmas = torch.tensor([self.sigma_max], device=device, dtype=torch.float32)
        else:
            sigmas = (
                self.sigma_max ** rho_inv
                + (steps / (num_steps - 1))
                * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
            ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


""" Samplers """

""" Many methods inspired by https://github.com/crowsonkb/k-diffusion/ """


class Sampler(nn.Module):
    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")


class KarrasSampler(Sampler):
    """https://arxiv.org/abs/2206.00364 algorithm 1"""

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(
        self, x: Tensor, fn: Callable, sigma: float, sigma_next: float, gamma: float
    ) -> Tensor:
        """Algorithm 2 (step)"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma
        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        # Evaluate ∂x/∂sigma at sigma_hat
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d
        # Second order correction
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / num_steps, sqrt(2) - 1),
            0.0,
        )
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(
                x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i]  # type: ignore # noqa
            )

        return x


class AEulerSampler(Sampler):
    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float]:
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Euler method
        x_next = x + d * (sigma_down - sigma)
        # Add randomness
        x_next = x_next + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x


class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    def __init__(self, rho: float = 1.0, num_resamples = None):
        super().__init__()
        self.rho = rho
        self.num_resamples = num_resamples

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)
        # Add randomness
        x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        # for i in range(num_steps - 1):
        for i in tqdm.tqdm(range(num_steps - 1)):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        # num_resamples: int,
    ) -> Tensor:
        x = sigmas[0] * torch.randn_like(source)

        # for i in range(num_steps - 1):
        for i in tqdm.tqdm(range(num_steps - 1)):
            # Noise source to current noise level
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(self.num_resamples):
                # Merge noisy source and current then denoise
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
                # Renoise if not last resample step
                if r < self.num_resamples - 1:
                    sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)

        return source * mask + x * ~mask


class MSDMSampler(Sampler):
    def __init__(self, num_resamples: int = 1, s_churn: float = 0.0):
        super().__init__()
        self.s_churn=s_churn 
        self.num_resamples=num_resamples

    def score_differential(self, x, sigma, denoise_fn):
        d = (x - denoise_fn(x, sigma=sigma)) / sigma 
        return d

    @torch.no_grad()
    def generate_track(
        self,
        denoise_fn: Callable,
        sigmas: torch.Tensor,
        noises: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = sigmas[0] * noises
        # _, num_sources, _  = x.shape    

        # Initialize default values
        source = torch.zeros_like(x) if source is None else source
        mask = torch.zeros_like(x) if mask is None else mask
        
        sigmas = sigmas.to(x.device)
        gamma = min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        
        # Iterate over all timesteps
        for i in tqdm.tqdm(range(len(sigmas) - 1)):
            sigma, sigma_next = sigmas[i], sigmas[i+1]

            # Noise source to current noise level
            noisy_source = source + sigma*torch.randn_like(source)
            
            for r in range(self.num_resamples):
                # Merge noisy source and current x
                x = mask*noisy_source + (1.0 - mask)*x 

                # Inject randomness
                sigma_hat = sigma * (gamma + 1)    
                x_hat = x + torch.randn_like(x) * (sigma_hat**2 - sigma**2)**0.5

                # Compute conditioned derivative
                d = self.score_differential(x=x_hat, sigma=sigma_hat, denoise_fn=denoise_fn)

                # Update integral
                x = x_hat + d*(sigma_next - sigma_hat)
                    
                # Renoise if not last resample step
                if r < self.num_resamples - 1:
                    x = x + torch.randn_like(x) * (sigma**2 - sigma_next**2)**0.5

        return mask*source + (1.0 - mask)*x


    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int ) -> Tensor:
        x = self.generate_track(fn,
                        sigmas=sigmas,
                        noises=noise,
                        )
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
    ) -> Tensor:
        x = self.generate_track(fn,
                        sigmas=sigmas,
                        noises=torch.randn_like(source),
                        source = source,
                        mask = mask.to(dtype=torch.float32),
                        )

        return x


""" loss function Classes """

class PerceptualLoss(nn.Module):
    def __init__(self, sample_rate: int = 22050, n_mels: int = 128):
        """
        Perceptual loss based on Mel spectrogram features.

        Args:
            sample_rate (int): Sample rate of the audio.
            n_mels (int): Number of Mel bands for the spectrogram.
        """
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2,
            power=2.0
        )

    def forward(self, x: Tensor, x_denoised: Tensor) -> Tensor:
        """
        Compute perceptual loss between input and denoised audio.

        Args:
            x (Tensor): Original input tensor (batch, channels, samples).
            x_denoised (Tensor): Denoised output tensor (batch, channels, samples).

        Returns:
            Tensor: Perceptual loss for each sample in the batch.
        """
        # Compute Mel spectrograms
        x_mel = self.mel_transform(x.squeeze(1))  # Remove channel dimension
        x_denoised_mel = self.mel_transform(x_denoised.squeeze(1))

        # Compute MSE loss between Mel spectrograms
        perceptual_loss = F.mse_loss(x_mel, x_denoised_mel, reduction="none")
        perceptual_loss = perceptual_loss.mean(dim=(-2, -1))  # Average across frequency and time

        return perceptual_loss


""" Diffusion Classes """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


class Diffusion(nn.Module):
    """Elucidated Diffusion: https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
        lambda_perceptual: float = 0.0, # Add lambda_perceptual as a parameter
        inpaint_mask_ratios: list[float] = None,
        pr_win_mul: list[float] = None,
    ):
        super().__init__()

        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold
        self.lambda_perceptual = lambda_perceptual  
        self.inpaint_mask_ratios = inpaint_mask_ratios
        self.pr_win_mul = pr_win_mul

        # Initialize PerceptualLoss if lambda_perceptual > 0
        self.perceptual_loss_fn = (
            PerceptualLoss() if lambda_perceptual > 0 else None
        )

    def get_scale_weights(self, sigmas: Tensor, x: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        sigmas_padded = sigmas.view(sigmas.shape[0], *([1] * (x.ndim - 1)))
        c_skip = (sigma_data ** 2) / (sigmas_padded ** 2 + sigma_data ** 2)
        c_out = (
            sigmas_padded * sigma_data * (sigma_data ** 2 + sigmas_padded ** 2) ** -0.5
        )
        c_in = (sigmas_padded ** 2 + sigma_data ** 2) ** -0.5
        c_noise = torch.log(sigmas) * 0.25
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch, device = x_noisy.shape[0], x_noisy.device

        assert exists(sigmas) ^ exists(sigma), "Either sigmas or sigma must be provided"

        # If sigma provided use the same for all batch items (used for sampling)
        if exists(sigma):
            sigmas = torch.full(size=(batch,), fill_value=sigma).to(device)

        assert exists(sigmas)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        # Dynamic thresholding
        if self.dynamic_threshold == 0.0:
            return x_denoised.clamp(-1.0, 1.0)
        elif self.dynamic_threshold == -1.0:
            return x_denoised
        else:
            # Find dynamic threshold quantile for each batch
            x_flat = x_denoised.view(x_denoised.shape[0], -1)
            scale = torch.quantile(x_flat.abs(), self.dynamic_threshold, dim=-1)
            # Clamp to a min of 1.0
            scale.clamp_(min=1.0)
            # Clamp all values and scale
            scale = scale.view(*scale.shape, *([1] * (x_denoised.ndim - scale.ndim)))
            x_denoised = x_denoised.clamp(-scale, scale) / scale
            return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def create_temporal_mask(self, like, mask_ratio):
        """
        Creates a temporal mask over the image-like spectrogram.
        - mask_ratio: percentage of the time axis to mask (e.g., 50%)
        - Assumes: First dim = Frequency (F), Second dim = Time (T)
        """
        _, F, T = like.shape  # Batch, Channels, Frequency, Time
        device = like.device
        mask = torch.ones_like(like, dtype=torch.bool)

        # Compute time range to mask (masking the last portion)
        t_mask = int(T * mask_ratio)  # Number of time steps to mask
        t_start = T - t_mask  # Start masking from this index

        # Apply mask to the last portion of the time axis
        mask[:, :, t_start:] = False  # Set masked area to False
        return mask

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch, device=device)
        sigmas_padded = sigmas.view(sigmas.shape[0], *([1] * (x.ndim - 1)))

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise
        # Add noise normally if no inpaint mask ratios are given
        if self.inpaint_mask_ratios:
        #     noise = default(noise, lambda: torch.randn_like(x))
        #     x_noisy = x + sigmas_padded * noise
        # else:
            # Choose a random mask ratio for each batch item
            mask_ratios = torch.tensor(
                [random.choice(self.inpaint_mask_ratios) for _ in range(batch)],
                device=device,
            )

            # # Generate the corresponding masks
            # masks = torch.stack([self.create_temporal_mask(x[i], mask_ratios[i]) for i in range(batch)])
            
            # # Create noise
            # noise = default(noise, lambda: torch.randn_like(x))

            # # Apply noise only to the last part of the masked image
            # # x_noisy = x.clone()
            # # x_noisy[~masks] += sigmas_padded[~masks] * noise[~masks]
            # x_noisy = x + (sigmas_padded * noise) * (~masks).float()


            # Apply masking logic to kwargs['mixture']
            if "mixture" in kwargs:
                mixture = kwargs["mixture"].clone()
                for i in range(batch):
                    mask_ratio = mask_ratios[i].item()
                    num_masks = random.choice(self.pr_win_mul)  # Choose how many times to apply mask ratio

                    # Determine how much of the last part to mask
                    total_mask_size = int(mixture.shape[-1] * (mask_ratio * num_masks))
                    if total_mask_size > 0:
                        start_idx = mixture.shape[-1] - total_mask_size
                        mixture[i, :, :, start_idx:] = 0.0 #noise[i, :, :, start_idx:]  # Replace with noise


                kwargs["mixture"] = mixture

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
            
        # if masks is not None:
        #     # Compute loss only on the masked regions
        #     losses = losses * (~masks).float()
        #     num_masked_elements = (~masks).float().sum(dim=list(range(1, masks.ndim)))  # Sum across all but batch dimension
        #     losses = losses.sum(dim=list(range(1, losses.ndim))) / (num_masked_elements + 1e-8)  # Avoid div by zero
        # else:
        #     # Compute full loss when no masks are applied
        #     losses = reduce(losses, "b ... -> b", "mean")        
          
        losses = reduce(losses, "b ... -> b", "mean")
        weigths = self.loss_weight(sigmas)
        losses = losses * weigths
        # loss = losses.mean()
        
        # Optionally add perceptual loss
        if self.perceptual_loss_fn is not None:
            perceptual_loss = self.perceptual_loss_fn(x, x_denoised)
            losses += self.lambda_perceptual * perceptual_loss

        # Final combined loss
        return losses.mean()


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = None,
        clamp: bool = True,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

    @torch.no_grad()
    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        if self.clamp:
            x = x.clamp(-1.0, 1.0)
        return x


class DiffusionInpainter(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        num_steps: int,
        # num_resamples: int,
        sampler: Sampler,
        sigma_schedule: Schedule,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        # self.num_resamples = num_resamples
        self.inpaint_fn = sampler.inpaint
        self.sigma_schedule = sigma_schedule

    @torch.no_grad()
    def forward(self, inpaint: Tensor, inpaint_mask: Tensor, **kwargs) -> Tensor:
        x = self.inpaint_fn(
            source=inpaint,
            mask=inpaint_mask,
            fn=lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs}),
            sigmas=self.sigma_schedule(self.num_steps, inpaint.device),
            num_steps=self.num_steps,
            # num_resamples=self.num_resamples,
        )
        return x


def sequential_mask(like: Tensor, start: int) -> Tensor:
    length, device = like.shape[2], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[:, :, start:] = torch.zeros((length - start,), device=device)
    return mask


class SpanBySpanComposer(nn.Module):
    def __init__(
        self,
        inpainter: DiffusionInpainter,
        *,
        num_spans: int,
    ):
        super().__init__()
        self.inpainter = inpainter
        self.num_spans = num_spans

    def forward(self, start: Tensor, keep_start: bool = False) -> Tensor:
        half_length = start.shape[2] // 2

        spans = list(start.chunk(chunks=2, dim=-1)) if keep_start else []
        # Inpaint second half from first half
        inpaint = torch.zeros_like(start)
        inpaint[:, :, :half_length] = start[:, :, half_length:]
        inpaint_mask = sequential_mask(like=start, start=half_length)

        for i in range(self.num_spans):
            # Inpaint second half
            span = self.inpainter(inpaint=inpaint, inpaint_mask=inpaint_mask)
            # Replace first half with generated second half
            second_half = span[:, :, half_length:]
            inpaint[:, :, :half_length] = second_half
            # Save generated span
            spans.append(second_half)

        return torch.cat(spans, dim=2)
