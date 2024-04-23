"""
This code is extended from guided_diffusion: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
Multi-modal guassian have been added, as well as zero-shot conditional generation.
"""

import enum
import math
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from . import dist_util


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
         
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x F x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
                
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(video_{t-1}, audio_{t-1} | video_t, aduio_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: {"video": [N x F x C x H x W], "audio": [N x C x T]}  at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
                 - 'model_outputs': the outputs of the model.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        B = x["video"].shape[0]
        assert t.shape == (B,)
        
        video_output, audio_output = model(x["video"], x["audio"], self._scale_timesteps(t), **model_kwargs) # when ddim, t is not mapped
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        def get_variance(model_output, x):
            if x.dim() == 3:
                dim=1
                
            elif x.dim() == 5 :
                dim=2
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                assert model_output.shape[dim] == x.shape[dim] * 2
                model_output, model_var_values = th.split(model_output, x.shape[dim], dim=dim)
                if self.model_var_type == ModelVarType.LEARNED:
                    model_log_variance = model_var_values
                    model_variance = th.exp(model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x.shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                    # The model_var_values is [-1, 1] for [min_var, max_var].
                    frac = (model_var_values + 1) / 2
                    model_log_variance = frac * max_log + (1 - frac) * min_log
                    model_variance = th.exp(model_log_variance)
            else:
                model_variance, model_log_variance = {
                    # for fixedlarge, we set the initial (log-)variance like so
                    # to get a better decoder log likelihood.
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                model_variance = _extract_into_tensor(model_variance, t, x.shape)
                model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
                )
                model_mean = model_output
            elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = process_xstart(model_output)

                else:
                    '''
                    if the model predicts the epsilon, pred xstart from predicted eps, then pred x_{t-1}
                    '''
                    pred_xstart = process_xstart(
                        self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                    )
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=x, t=t
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
            )
            return model_mean, model_variance, model_log_variance, pred_xstart
     
        video_mean, video_variance, video_log_variance, pred_video_xstart = get_variance(video_output, x["video"])
        audio_mean, audio_variance, audio_log_variance, pred_audio_xstart = get_variance(audio_output, x["audio"])

        return {
            "mean": {"video": video_mean, "audio": audio_mean},
            "variance": {"video":video_variance, "audio": audio_variance},
            "log_variance": {"video":video_log_variance, "audio": audio_log_variance},
            "pred_xstart": {"video": pred_video_xstart, "audio":pred_audio_xstart},
            "model_predict":{"video":video_output, "audio": audio_output}
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        '''
        scale step t
        '''
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor dict at x_{t-1}: {"video":[N,F,C,H,W]; "audio":[N,C,T]}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_start': a prediction of x_0.
                 - 'pred_noise': a prediction of epsilon.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        video_noise = th.randn_like(x["video"])
        audio_noise = th.randn_like(x["audio"])
        
        video_nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x["video"].shape) - 1)))
        )  # no noise when t == 0

        audio_nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x["audio"].shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
   
        video_sample = out["mean"]["video"] + video_nonzero_mask * th.exp(0.5 * out["log_variance"]["video"]) * video_noise
        audio_sample = out["mean"]["audio"] + audio_nonzero_mask * th.exp(0.5 * out["log_variance"]["audio"]) * audio_noise

        return {"sample": {"video": video_sample, "audio":  audio_sample}, \
            "pred_start": {"video": out["pred_xstart"]["video"], "audio": out["pred_xstart"]["audio"]},
            "pred_noise":{"video": out["model_predict"]["video"], "audio": out["model_predict"]["audio"]}}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, {'video':(N, F, C, H, W), 'audio':(N, C, T)}
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1]. 
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress
        ):
            final = sample
        return final


    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
      
        if device is None:
            device = dist_util.dev()
                
        video = th.randn(*shape["video"], device='cpu')
        video = video.to(device)

        audio = th.randn(*shape["audio"], device='cpu')
        audio = audio.to(device)
        x = {"video": video, "audio":audio}
        indices = list(range(self.num_timesteps))[::-1] # 0 to 999
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        for i in indices:
            cond = None
            timestep = self.timestep_map[i]
            if cond_fn is not None: 
                cond= cond_fn
            

            t = th.tensor([i] * shape["video"][0], device=device)
                
            with th.no_grad():
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                yield out["sample"]
                x = out["sample"]
    
    def conditional_p_sample_loop(
        self,
        model,
        shape,
        use_fp16,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        class_scale=0.
    ):
        """
        Zero-shot conditional generation of samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, {"video":(N, F, C, H, W), "audio":(N,C,T)}.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param class_scale: if 0, replacement method is used. Otherwise, the gradient based method is used. 
        :return: a non-differentiable batch of samples.
        """
        final = None
        if class_scale == 0:
            conditional_p_sample_loop_progressive_func = self.conditional_p_sample_loop_progressive_unscale
        else:
            conditional_p_sample_loop_progressive_func = self.conditional_p_sample_loop_progressive_scale

        for sample in conditional_p_sample_loop_progressive_func(
            model,
            shape,
            use_fp16,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            class_scale=class_scale
        ):
            final = sample

        return final


    def conditional_p_sample_loop_progressive_unscale(
        self,
        model,
        shape,
        use_fp16,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        class_scale=0.0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as conditional_p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
      
        if device is None:
            device = dist_util.dev()
        
        if noise is None:
            video = th.randn(*shape["video"], device='cpu')
            video = video.to(device)

            audio = th.randn(*shape["audio"], device='cpu')
            audio = audio.to(device)
            noise = {"video": video, "audio":audio}
       
        x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        video_condition = None
        audio_condition= None

        if "video" in model_kwargs.keys():
            video_condition = model_kwargs.pop("video")

        if "audio" in model_kwargs.keys():
            audio_condition = model_kwargs.pop("audio")

        for i in indices:
            cond = None
            timestep = self.timestep_map[i]
            if cond_fn is not None: 
                cond = cond_fn

            t = th.tensor([i] * shape["video"][0], device=device)
           
            if video_condition is not None:
                x["video"] = self.q_sample(video_condition, t, noise = noise["video"])
            
            if audio_condition is not None:
                x["audio"] = self.q_sample(audio_condition, t, noise = noise["audio"])

                            
            with th.no_grad():
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                yield out["sample"]
                x = out["sample"]   
                
    def conditional_p_sample_loop_progressive_scale(
        self,
        model,
        shape,
        use_fp16,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        class_scale=3.0,
        
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as conditional_p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
      
        if device is None:
            device = dist_util.dev()
        
        if noise is None:
            video = th.randn(*shape["video"], device='cpu')
            video = video.to(device)

            audio = th.randn(*shape["audio"], device='cpu')
            audio = audio.to(device)
            noise = {"video": video, "audio":audio}
       
        x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        video_condition = None
        audio_condition= None

        if "video" in model_kwargs.keys():
            video_condition = model_kwargs.pop("video")

        if "audio" in model_kwargs.keys():
            audio_condition = model_kwargs.pop("audio")
        
        for i in indices:
            cond = None
            timestep = self.timestep_map[i]
            if cond_fn is not None: 
                cond = cond_fn

            t = th.tensor([i] * shape["video"][0], device = device)
            # first get unconditional generation results
            
            if video_condition is not None:
                condition = "video"
                target = "audio"
                x[condition] = self.q_sample(video_condition, t, noise = noise[condition])
                previous_step_condition = self.q_sample(video_condition, t-1, noise = noise[condition])
            if audio_condition is not None:
                condition = "audio"
                target = "video"
                x[condition] = self.q_sample(audio_condition, t, noise = noise[condition])
                previous_step_condition = self.q_sample(audio_condition, t-1, noise = noise[condition])
            
            with th.enable_grad():                
                none_zero_mask = (t != 0).float().view(-1, *([1] * (len(x[target].shape) - 1)))
                x[target] = x[target].detach().requires_grad_()
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                
                previous_step_pred = out["sample"]
                
                loss = mean_flat((previous_step_pred[condition] - previous_step_condition) ** 2)
                loss_scale = 1.
                if use_fp16 == True:
                    loss_scale = 2 ** 20
                grad = th.autograd.grad(loss.mean()* loss_scale , x[target])[0]
                # print(f"!!!!!!!!!!!!!!!grad:{grad.sum()}")
                x[target] = previous_step_pred[target] - none_zero_mask * grad* class_scale * self.sqrt_alphas_cumprod[i]              
                            
            yield x
 
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = {}
        eps["video"] = self._predict_eps_from_xstart(x["video"], t, out["pred_xstart"]["video"])
        eps["audio"] = self._predict_eps_from_xstart(x["audio"], t, out["pred_xstart"]["audio"])

        alpha_bar = {}
        alpha_bar["video"] = _extract_into_tensor(self.alphas_cumprod, t, x["video"].shape)
        alpha_bar["audio"] = _extract_into_tensor(self.alphas_cumprod, t, x["audio"].shape)

        alpha_bar_prev = {}
        alpha_bar_prev["video"] = _extract_into_tensor(self.alphas_cumprod_prev, t, x["video"].shape)
        alpha_bar_prev["audio"] = _extract_into_tensor(self.alphas_cumprod_prev, t, x["audio"].shape)
        
        sigma = {}
        sigma["video"] = (
            eta * th.sqrt((1 - alpha_bar_prev["video"]) / (1 - alpha_bar["video"]))
            * th.sqrt(1 - alpha_bar["video"] / alpha_bar_prev["video"])
        )
        sigma["audio"] = (
            eta * th.sqrt((1 - alpha_bar_prev["audio"]) / (1 - alpha_bar["audio"]))
            * th.sqrt(1 - alpha_bar["audio"] / alpha_bar_prev["audio"])
        )

        # Equation 12.
        noise = {}
        noise["video"] = th.randn_like(x["video"])
        noise["audio"] = th.randn_like(x["audio"])

        mean_pred = {}
        mean_pred["video"] = (
            out["pred_xstart"]["video"] * th.sqrt(alpha_bar_prev["video"])
            + th.sqrt(1 - alpha_bar_prev["video"] - sigma["video"] ** 2) * eps["video"]
        )
        mean_pred["audio"] = (
            out["pred_xstart"]["audio"] * th.sqrt(alpha_bar_prev["audio"])
            + th.sqrt(1 - alpha_bar_prev["audio"] - sigma["audio"] ** 2) * eps["audio"]
        )

        nonzero_mask = {}
        nonzero_mask["video"] = (
            (t != 0).float().view(-1, *([1] * (len(x["video"].shape) - 1)))
        )  # no noise when t == 0
        nonzero_mask["audio"] = (
            (t != 0).float().view(-1, *([1] * (len(x["audio"].shape) - 1)))
        )  # no noise when t == 0

        sample = {}
        sample["video"] = mean_pred["video"] + nonzero_mask["video"] * sigma["video"] * noise["video"]
        sample["audio"] = mean_pred["audio"] + nonzero_mask["audio"] * sigma["audio"] * noise["audio"]

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = {}
        eps["video"] = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["video"].shape) * x["video"]
            - out["video"]["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["video"].shape)

        eps["audio"] = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["audio"].shape) * x["audio"]
            - out["audio"]["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["audio"].shape)

        alpha_bar_next = {}
        alpha_bar_next["video"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["video"].shape)
        alpha_bar_next["audio"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["audio"].shape)

        # Equation 12. reversed
        mean_pred = {}
        mean_pred["video"] = (
            out["pred_xstart"]["video"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps["video"]
        )
        mean_pred["audio"] = (
            out["pred_xstart"]["audio"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps["audio"]
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta
        ):
            final = sample
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = dist_util.dev()
       
  
        video = th.randn(*shape["video"], device='cpu')
        video = video.to(device)

        audio = th.randn(*shape["audio"], device='cpu')
        audio = audio.to(device)
        x = {"video": video, "audio":audio}

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape["video"][0], device=device)
            cond=None
            timestep = self.timestep_map[i]
            if cond_fn is not None:
                cond = cond_fn

            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out["sample"]
                x = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        video_true_mean, _, video_true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start["video"], x_t=x_t["video"], t=t
        )
        audio_true_mean, _, audio_true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start["audio"], x_t=x_t["audio"], t=t
        )
        true_mean = {"video": video_true_mean, "audio": audio_true_mean}
        true_log_variance_clipped = {"video": video_true_log_variance_clipped, "audio": audio_true_log_variance_clipped}
        
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
     
        kl = {}
        decoder_nll={}
        output = {}
        for key in ["video", "audio"]:
            kl[key] = normal_kl(
            true_mean[key], true_log_variance_clipped[key], out["mean"][key], out["log_variance"][key]
        )
            kl[key] = mean_flat(kl[key]) / np.log(2.0)

            decoder_nll[key] = -discretized_gaussian_log_likelihood(
            x_start[key], means=out["mean"][key], log_scales=0.5 * out["log_variance"][key]
        )
            assert decoder_nll[key].shape == x_start[key].shape
            decoder_nll[key] = mean_flat(decoder_nll[key]) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            output[key] = th.where((t == 0), decoder_nll[key], kl[key])
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def predict_image_qt_t_step(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Predict Image at t_th step with model
        :param model: the model to predict_image.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        return x_t

    def multimodal_training_losses(self, model, x_start, t,  model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        video_start = x_start['video']
        audio_start = x_start['audio']
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise ={"video":th.randn_like(video_start),\
                "audio":th.randn_like(audio_start)}
        
        #0 means t_th step, 1 means the audio gives groundtruth, 2 means the video gives the groundtruth
        # condition_index = x_start["condition"]  
        video_t = self.q_sample(video_start, t, noise = noise["video"])
        audio_t = self.q_sample(audio_start, t, noise = noise["audio"])
  
        video_output, audio_output = model(video_t, audio_t, self._scale_timesteps(t),  **model_kwargs)
     
        video_loss = {}
        audio_loss = {}
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
                
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                  
                
                video_output, video_var_values = th.split(video_output, video_start.shape[2], dim=2)
                audio_output, audio_var_values = th.split(audio_output, audio_start.shape[1], dim=1)
                    # Learn the variance using the variational bound, but don't let
                    # it affect our mean prediction.
                video_frozen_out = th.cat([video_output.detach(), video_var_values], dim=2)
                audio_frozen_out = th.cat([audio_output.detach(), audio_var_values], dim=1)
                frozen_out = {"video": video_frozen_out, "audio": audio_frozen_out}
                x_t = {"video": video_t, "audio": audio_t}
                vb_loss = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: [r["video"], r["audio"]],
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                video_loss["vb"] = vb_loss["video"]
                audio_loss["vb"] = vb_loss["audio"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    video_loss["vb"] *= self.num_timesteps / 1000.0
                    audio_loss["vb"] *= self.num_timesteps / 1000.0
         
            video_target = {
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=video_start, x_t=video_t, t=t
                    )[0],
                    ModelMeanType.START_X: video_start,
                    ModelMeanType.EPSILON: noise["video"],   # noise
                }[self.model_mean_type]
            audio_target = {
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=audio_start, x_t=audio_t, t=t
                    )[0],
                    ModelMeanType.START_X: audio_start,
                    ModelMeanType.EPSILON: noise["audio"],   # noise
                }[self.model_mean_type]     
              
            video_loss["mse"] = mean_flat((video_target - video_output) ** 2)    
            audio_loss["mse"] = mean_flat((audio_target - audio_output) ** 2)
            
        term = {"loss":0}
 
        for key in video_loss.keys():
            term[f"{key}_video"] = video_loss[key]
            term[f"{key}_audio"] =  audio_loss[key]
            # term[f"{key}_all"] = video_mask * video_loss[key] + audio_mask * audio_loss[key]
            term["loss"] += term[f"{key}_video"] + term[f"{key}_audio"]
  

        return term


    def _motion_variance(self, predict, target):
       
        assert predict.shape==target.shape
        predict_motion = predict[:,1:,...]-predict[:,:-1,...]
        target_motion = target[:,1:,...]-target[:,:-1,...]
        return  0.05*mean_flat((predict_motion - target_motion) ** 2) 
        
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
   
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
