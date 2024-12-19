import torch
import torch.nn.functional as F
import gc
from typing import Optional

from diffusion_core.utils.class_registry import ClassRegistry
from diffusion_core.p2p.seq_aligner import get_mapper
from diffusion_core.custom_forwards.unet_sd import unet_forward
from diffusion_core.guiders.scale_schedulers import last_steps, first_steps
from diffusers.image_processor import IPAdapterMaskProcessor
from typing import Callable, List, Optional, Tuple, Union
opt_registry = ClassRegistry()


class BaseGuider:
    def __init__(self):
        self.clear_outputs()
    
    @property
    def grad_guider(self):
        return hasattr(self, 'grad_fn')
    
    def __call__(self, data_dict):
        if self.grad_guider:
            return self.grad_fn(data_dict)
        else:
            return self.calc_energy(data_dict)   
        
    def clear_outputs(self):
        if not self.grad_guider:
            self.output = self.single_output_clear()
    
    def single_output_clear(self):
        raise NotImplementedError()

        
@opt_registry.add_to_registry('cfg')
class ClassifierFreeGuidance(BaseGuider):
    def __init__(self, is_source_guidance=False):
        self.is_source_guidance = is_source_guidance
    
    def grad_fn(self, data_dict):
        prompt_unet = data_dict['src_prompt_unet'] if self.is_source_guidance else data_dict['trg_prompt_unet']
        return prompt_unet - data_dict['uncond_unet']


@opt_registry.add_to_registry('latents_diff')
class LatentsDiffGuidance(BaseGuider):
    """
    \| z_t* - z_t \|^2_2
    """
    def grad_fn(self, data_dict):
        return 2 * (data_dict['latent'] - data_dict['inv_latent'])

    
@opt_registry.add_to_registry('midu-l2')
class MidUL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_trg', 'inv_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['midu-l2_cur_trg'] - data_dict['midu-l2_inv_inv'], 2))
    
    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output 
        model.unet.mid_block.register_forward_hook(hook_fn)
    
    def single_output_clear(self):
        None
        
    
@opt_registry.add_to_registry('midu-src-l2')
class MidUL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['midu-src-l2_cur_inv'] - data_dict['midu-src-l2_inv_inv'], 2))
    
    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output
        model.unet.mid_block.register_forward_hook(hook_fn)
    
    def single_output_clear(self):
        None


class AttentionPatchGuider(BaseGuider):
    patched = False
    def single_output_clear(self):
        return {
            "down_attn_cross": [], "mid_attn_cross": [], "up_attn_cross": [],
            "down_attn_self":  [], "mid_attn_self":  [], "up_attn_self":  [],

            "down_query_cross": [], "mid_query_cross": [], "up_query_cross": [],
            "down_query_self": [], "mid_query_self": [], "up_query_self": [],

            "down_key_self": [], "mid_key_self": [], "up_key_self": [],
            "down_key_cross": [], "mid_key_cross": [], "up_key_cross": [],

            "down_value_self": [], "mid_value_self": [], "up_value_self": [],
            "down_value_cross": [], "mid_value_cross": [], "up_value_cross": [],
        }
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        if guider_self.patched:
            return
        
        guider_self.patched = True
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_cross = encoder_hidden_states is not None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                
                if getattr(guider_self, 'attn_cross', False) and is_cross:
                    guider_self.output[f"{place_unet}_attn_cross"].append(attention_probs)
                elif getattr(guider_self, 'attn_self', False) and not is_cross:
                    guider_self.output[f"{place_unet}_attn_self"].append(attention_probs)
                
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet):
            patchers = {
                'attn_self': guider_self.attn_self,
                'attn_cross': guider_self.attn_cross
            }
                
            
            if 'Attention' in module.__class__.__name__:
                for attr, value in patchers.items():
                    if not value:
                        continue
                    if not hasattr(module, attr):
                        setattr(module, attr, value)
                    module.forward = new_forward_info(module, place_in_unet)    
                if patch_self:
                    if not hasattr(module, 'attn_self'):
                        module.attn_self = True
                    module.forward = new_forward_info(module, place_in_unet)
                elif patch_cross:
                    if not hasattr(module, 'attn_cross'):
                        module.attn_cross = True
                    module.forward = new_forward_info(module, place_in_unet)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_, place_in_unet)
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down")
            elif "up" in name:
                register_attn(net, "up")
            elif "mid" in name:
                register_attn(net, "mid")

                
@opt_registry.add_to_registry('src_attn_map_l2_refactored')
class AttnMapL2EnergyGuiderR(AttentionPatchGuider):
    forward_hooks = ['cur_inv', 'inv_inv']
    def __init__(self, attn_self=False, attn_cross=True):
        super().__init__()
        self.attn_self = attn_self
        self.attn_cross = attn_cross
        
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['src_attn_map_l2_refactored_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['src_attn_map_l2_refactored_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        return result    


@opt_registry.add_to_registry('src_attn_map_l2')
class AttnMapL2EnergyGuider(AttentionPatchGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['src_attn_map_l2_cur_inv'].items():
            if data:
                for elem_idx, elem in enumerate(data):
                    result += torch.mean(
                        torch.pow(
                            elem - data_dict['src_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                        )
                    )
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_cross = encoder_hidden_states is not None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_cross:
                    guider_self.output[f"{place_unet}_attn_cross"].append(attention_probs)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info(module, place_in_unet)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_, place_in_unet)
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down")
            elif "up" in name:
                register_attn(net, "up")
            elif "mid" in name:
                register_attn(net, "mid")

                
@opt_registry.add_to_registry('features_map_l2')
class FeaturesMapL2EnergyGuider(BaseGuider):
    def __init__(self, block='up'):
        assert block in ['down', 'up', 'mid', 'whole']
        self.block = block
        
    patched = True
    forward_hooks = ['cur_trg', 'inv_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['features_map_l2_cur_trg'] - data_dict['features_map_l2_inv_inv'], 2))
    
    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output 
        if self.block == 'mid':
            model.unet.mid_block.register_forward_hook(hook_fn)
        elif self.block == 'up':
            model.unet.up_blocks[1].resnets[1].register_forward_hook(hook_fn)
        elif self.block == 'down':
            model.unet.down_blocks[1].resnets[1].register_forward_hook(hook_fn)
    
    def single_output_clear(self):
        None
    
    
@opt_registry.add_to_registry('self_attn_map_l2')
class SelfAttnMapL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']    
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['self_attn_map_l2_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)
                print("self attn map l2!!!!!!!!!!!!!!!!!")
                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        # def register_attn(module, place_in_unet):
        #     if 'Attention' in module.__class__.__name__:
        #         module.forward = new_forward_info(module, place_in_unet)
                
        #     elif hasattr(module, 'children'):
        #         for module_ in module.children():
        #             register_attn(module_, place_in_unet)
        
        # sub_nets = model.unet.named_children()
        # for name, net in sub_nets:
        #     if "down" in name:
        #         register_attn(net, "down")
        #     if "up" in name:
        #         register_attn(net, "up")
        #     if "mid" in name:
        #         register_attn(net, "mid")
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])


@opt_registry.add_to_registry('src_attn_map_appearance')
class AttnMapAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['inv_inv', 'cur_inv'] 
    def single_output_clear(self):
        return {
            "up_cross": [],
            "features": None
        }
    
    def __normalize(self, x):
        mn = torch.min(x, dim=0, keepdim=True)[0]
        mx = torch.max(x, dim=0, keepdim=True)[0]
        normalized = (x - mn) / (mx - mn)
        return normalized
    
    def features_by_shape(self, features, shape):
        shape = shape.reshape(1, 1, features.shape[-2], features.shape[-1], -1)
        features = features.reshape(*features.shape, 1)
        return shape * features
    
    def calc_energy(self, data_dict):
        shape_orig = data_dict['src_attn_map_appearance_inv_inv']['up_cross'][4].detach().mean(dim=0)[..., 0]
        shape_orig = torch.nn.functional.interpolate(shape_orig.reshape(1, 1, 32, 32), size=(64, 64))

        features_orig = data_dict['src_attn_map_appearance_inv_inv']['features']
        features_cur = data_dict['src_attn_map_appearance_cur_inv']['features']

        result = torch.mean(torch.abs(self.features_by_shape((features_orig - features_cur), shape_orig)))

        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info_attn_map(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_cross = encoder_hidden_states is not None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_cross:
                    guider_self.output[f"{place_unet}_{'cross' if is_cross else 'self'}"].append(attention_probs)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info_attn_map(module, place_in_unet)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_, place_in_unet)
        
        def new_forward_info_features(self):
            def patched_forward(input):
                out = F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
                guider_self.output["features"] = out
                return out
            return patched_forward
        
        register_attn(model.unet.up_blocks, 'up')
        model.unet.conv_norm_out.forward = new_forward_info_features(model.unet.conv_norm_out)


@opt_registry.add_to_registry('src_cross_attn_map_l2')
class CrossAttnMapL2EnergyGuider(BaseGuider):
    def __init__(self, do_mapper=True):
        self.do_mapper = do_mapper

        self.mapper_orig = None
        self.mapper_cur = None
        self.output = self.single_output_clear()
        
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']    
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['src_cross_attn_map_l2_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['src_cross_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        return result
    
    def train(self, params):
        """
        For this guider we need tokens from source prompt, that do not appear in target prompt
        """
        if self.do_mapper:
            mapper, _ = get_mapper(
                params['trg_prompt'],
                params['inv_prompt'],
                params['model'].tokenizer
            )
            self.mapper_orig = mapper != -1
            self.mapper_cur = mapper[mapper != -1]
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_cross = encoder_hidden_states is not None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_cross:
                    guider_self.output[f"{place_unet}_{'cross' if is_cross else 'self'}"].append(attention_probs[:, :, guider_self.mapper_orig])
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info(module, place_in_unet)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_, place_in_unet)
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down")
            elif "up" in name:
                register_attn(net, "up")
            elif "mid" in name:
                register_attn(net, "mid")

    
@opt_registry.add_to_registry('self_attn_map_l2_appearance')
class SelfAttnMapL2withAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

    def single_output_clear(self):
        return {
            "down_self":  [], 
            "mid_self":  [], 
            "up_self":  [],
            "features": None
        }
    
    def calc_energy(self, data_dict): 
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        for unet_place in unet_places:
            data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
            for elem_idx, elem in enumerate(data):
                self_attn_result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        
        features_orig = data_dict['self_attn_map_l2_appearance_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cur_inv']['features']
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
        
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                ip_adapter_masks: Optional[torch.Tensor] = None
            ):
                if encoder_hidden_states is None: #self-attn 
                    residual = hidden_states
                    if self.spatial_norm is not None:
                        hidden_states = self.spatial_norm(hidden_states, temb)

                    input_ndim = hidden_states.ndim
                    if input_ndim == 4:
                        batch_size, channel, height, width = hidden_states.shape
                        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                    batch_size, sequence_length, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                    )
                    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                    if self.group_norm is not None:
                        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    query = self.to_q(hidden_states) 
                    encoder_hidden_states = hidden_states
                    key = self.to_k(encoder_hidden_states)
                    value = self.to_v(encoder_hidden_states)

                    query = self.head_to_batch_dim(query)
                    key = self.head_to_batch_dim(key)
                    value = self.head_to_batch_dim(value)
                    attention_probs = self.get_attention_scores(query, key, attention_mask)
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)   
                    hidden_states = torch.bmm(attention_probs, value)
                    hidden_states = self.batch_to_head_dim(hidden_states)

                    # linear proj
                    hidden_states = self.to_out[0](hidden_states)
                    # dropout
                    hidden_states = self.to_out[1](hidden_states)

                    if input_ndim == 4:
                        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                    if self.residual_connection:
                        hidden_states = hidden_states + residual

                    hidden_states = hidden_states / self.rescale_output_factor
                    return hidden_states
                else: # cross_attention 
                    residual = hidden_states
                    if self.spatial_norm is not None:
                        hidden_states = self.spatial_norm(hidden_states, temb)

                    input_ndim = hidden_states.ndim

                    if input_ndim == 4:
                        batch_size, channel, height, width = hidden_states.shape
                        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                    batch_size, sequence_length, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                    )

                    if attention_mask is not None:
                        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                        attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

                    if self.group_norm is not None:
                        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    query = self.to_q(hidden_states)
                    if encoder_hidden_states.shape[1] == self.processor.full_emb_size:
                        if encoder_hidden_states is None:
                            encoder_hidden_states = hidden_states
                        else:
                            end_pos = encoder_hidden_states.shape[1] - self.processor.num_tokens
                            encoder_hidden_states, ip_hidden_states = (
                                encoder_hidden_states[:, :end_pos, :],
                                encoder_hidden_states[:, end_pos:, :],
                            )
                            if self.norm_cross:
                                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                        key = self.to_k(encoder_hidden_states)
                        value = self.to_v(encoder_hidden_states)

                        inner_dim = key.shape[-1]
                        head_dim = inner_dim // self.heads

                        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        hidden_states = F.scaled_dot_product_attention(
                            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                        )

                        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                        hidden_states = hidden_states.to(query.dtype)
                        ip_key = self.processor.to_k_ip(ip_hidden_states.clone())
                        ip_value = self.processor.to_v_ip(ip_hidden_states.clone())
                        ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                        with torch.no_grad():
                            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)

                        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                        ip_hidden_states = ip_hidden_states.to(query.dtype)

                        if ip_adapter_masks is not None:    #with ip_adapter mask
                            print("with mask!!!!!!!!!!!!")
                            if not isinstance(ip_adapter_masks, List):
                                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
   
                            for mask in ip_adapter_masks:
                                mask_downsample = IPAdapterMaskProcessor.downsample(
                                    mask[:, 0, :, :], # now we have mask only for 1 obj
                                    batch_size,
                                    ip_hidden_states.shape[1],
                                    ip_hidden_states.shape[2],
                                )
                    
                            mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                            hidden_states = hidden_states + self.processor.scale * ip_hidden_states * mask_downsample
                        else:
                            hidden_states = hidden_states + self.processor.scale * ip_hidden_states

                        hidden_states = self.to_out[0](hidden_states)
                        hidden_states = self.to_out[1](hidden_states)

                        if input_ndim == 4:
                            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                        if self.residual_connection:
                            hidden_states = hidden_states + residual

                        hidden_states = hidden_states / self.rescale_output_factor

                        return hidden_states  
                    else:
                        #without image cross-attention 
                        if encoder_hidden_states is None:
                            encoder_hidden_states = hidden_states
                        else:
                            if self.norm_cross:
                                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                        key = self.to_k(encoder_hidden_states)
                        value = self.to_v(encoder_hidden_states)

                        inner_dim = key.shape[-1]
                        head_dim = inner_dim // self.heads

                        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                        hidden_states = F.scaled_dot_product_attention(
                            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                        )

                        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                        hidden_states = hidden_states.to(query.dtype)
                        # linear proj
                        hidden_states = self.to_out[0](hidden_states)
                        # dropout
                        hidden_states = self.to_out[1](hidden_states)

                        if input_ndim == 4:
                            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                        if self.residual_connection:
                            hidden_states = hidden_states + residual

                        hidden_states = hidden_states / self.rescale_output_factor

                        return hidden_states           
            return patched_forward
        
        # def register_attn(module, place_in_unet):
        #     if 'Attention' in module.__class__.__name__:
        #         module.forward = new_forward_info(module, place_in_unet)
        #     elif hasattr(module, 'children'):
        #         for module_ in module.children():
        #             register_attn(module_, place_in_unet)
        # sub_nets = model.unet.named_children()
        # for name, net in sub_nets:
        #     if "down" in name:
        #         register_attn(net, "down")
        #     if "up" in name:
        #         register_attn(net, "up")
        #     if "mid" in name:
        #         register_attn(net, "mid")
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        # print("self_attn_layers_num[0]",self_attn_layers_num[0])
        # print("self_attn_layers_num[1]",self_attn_layers_num[1])
        # print("self_attn_layers_num[2]",self_attn_layers_num[2])
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
        
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.unet.up_blocks[-1].register_forward_hook(hook_fn) 
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
