import torch
from typing import Optional, Tuple, Union, Dict, Any
from diffusion_core.utils import checkpoint_forward

@checkpoint_forward
def unet_down_forward(downsample_block, sample, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )
    else:
        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
    return sample, res_samples


@checkpoint_forward
def unet_mid_forward(mid_block, sample, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
    sample = mid_block(
        sample,
        emb,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    return sample


@checkpoint_forward
def unet_up_forward(upsample_block, sample, emb, res_samples, encoder_hidden_states, cross_attention_kwargs,
                    upsample_size, attention_mask):
    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
        sample = upsample_block(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=res_samples,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            upsample_size=upsample_size,
            attention_mask=attention_mask,
        )
    else:
        sample = upsample_block(
            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
        )
    return sample


def unet_forward(
        model,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond=None,
        controlnet_conditioning_scale=1.,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
):
    if controlnet_cond is not None:
        down_block_additional_residuals, mid_block_additional_residual = model.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        down_block_additional_residuals = [
            down_block_res_sample * controlnet_conditioning_scale
            for down_block_res_sample in down_block_additional_residuals
        ]
        mid_block_additional_residual *= controlnet_conditioning_scale
    else:
        down_block_additional_residuals = None
        mid_block_additional_residual = None

    default_overall_up_factor = 2 ** model.unet.num_upsamplers
    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        logger.info("Forward upsample size to force interpolation output size.")
        forward_upsample_size = True

    # prepare attention_mask
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)
        #print("attention_mask.shape2",  attention_mask.shape)

    # 0. center input if necessary
    if model.unet.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = model.unet.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=model.unet.dtype)

    emb = model.unet.time_embedding(t_emb, timestep_cond)

    if model.unet.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if model.unet.config.class_embed_type == "timestep":
            class_labels = model.unet.time_proj(class_labels)

        class_emb = model.unet.class_embedding(class_labels).to(dtype=model.unet.dtype)
        emb = emb + class_emb

    # 2. pre-process
    sample = model.unet.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in model.unet.down_blocks:
        sample, res_samples = unet_down_forward(downsample_block, sample, emb, encoder_hidden_states, attention_mask,
                                                cross_attention_kwargs)

        down_block_res_samples += res_samples

    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if model.unet.mid_block is not None:
        sample = unet_mid_forward(model.unet.mid_block, sample, emb, encoder_hidden_states, attention_mask,
                                  cross_attention_kwargs)

    if mid_block_additional_residual is not None:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(model.unet.up_blocks):
        is_final_block = i == len(model.unet.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        sample = unet_up_forward(upsample_block, sample, emb, res_samples, encoder_hidden_states,
                                 cross_attention_kwargs, upsample_size, attention_mask)

    # 6. post-process
    if model.unet.conv_norm_out:
        sample = model.unet.conv_norm_out(sample)
        sample = model.unet.conv_act(sample)
    sample = model.unet.conv_out(sample)

    # if not return_dict:
    #     return (sample,)

    return sample
