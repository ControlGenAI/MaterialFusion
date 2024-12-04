import enum
import torch
import numpy as np
from typing import Optional

from diffusion_core.guiders.scale_schedulers import last_steps_linear, first_steps_linear
from diffusion_core.utils import ClassRegistry

noise_rescales = ClassRegistry()


class RescaleType(enum.Enum):
    UPPER = 0
    RANGE = 1


class BaseNoiseRescaler:
    def __init__(self, noise_rescale_setup):
        if isinstance(noise_rescale_setup, float):
            self.upper_bound = noise_rescale_setup
            self.rescale_type = RescaleType.UPPER
        elif len(noise_rescale_setup) == 2:
            self.upper_bound, self.upper_bound = noise_rescale_setup
            self.rescale_type = RescaleType.RANGE
        else:
            raise TypeError('Incorrect noise_rescale_setup type: possible types are float, tuple(float, float)')
            
    def __call__(self, noises, index):
        if 'other' not in noises:
            return {k: 1. for k in noises}
        rescale_dict = self._rescale(noises, index)
        return rescale_dict
    
    def _rescale(self, noises, index):
        raise NotImplementedError('')

        
@noise_rescales.add_to_registry('identity_rescaler')
class IdentityRescaler:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, noises, index):
        return {k: 1. for k in noises}
    

@noise_rescales.add_to_registry('multiply_other_on_cfg_norm')
class MultNoiseRescaler(BaseNoiseRescaler):
    def __init__(self, noise_rescale_setup, scale_decrease_steps: Optional[int] = None):
        super().__init__(noise_rescale_setup)
        assert isinstance(noise_rescale_setup, float), 'incorrect type of noise_rescale_setup'

        if scale_decrease_steps is not None:
            self.scales = last_steps_linear(noise_rescale_setup, scale_decrease_steps)
        else:
            self.scales = [noise_rescale_setup] * 50
        
    def _rescale(self, noises, index):
        cfg_noise_norm = torch.norm(noises['cfg']).item()
        other_noise_norm = torch.norm(noises['other']).item()
        
        ratio = other_noise_norm / cfg_noise_norm if cfg_noise_norm != 0 else 1.
        if other_noise_norm != 0.:
            other_scale = self.scales[index] / ratio
        else:
            other_scale = 1.
        
        answer = {
            'cfg': 1.,
            'uncond': 1.,
            'other': other_scale
        }
        return answer

    
@noise_rescales.add_to_registry('upper_other_on_cfg_norm')
class UpperNoiseRescaler(BaseNoiseRescaler):
    def __init__(self, noise_rescale_setup):
        super().__init__(noise_rescale_setup)
        assert isinstance(noise_rescale_setup, float), 'incorrect type of noise_rescale_setup'
        self.upper_bound = noise_rescale_setup
        
    def __call__(self, noises, index):
        cfg_noise_norm = torch.norm(noises['cfg']).item()
        other_noise_norm = torch.norm(noises['other']).item()
        
        ratio = other_noise_norm / cfg_noise_norm if cfg_noise_norm != 0 else 1.
        ratio_clipped = np.clip(ratio, ratio - 1, self.upper_bound)
        if other_noise_norm != 0.:
            other_scale = ratio_clipped / ratio
        else:
            other_scale = 1.
        
        answer = {
            'cfg': 1.,
            'uncond': 1.,
            'other': other_scale
        }
        return answer
        
        
@noise_rescales.add_to_registry('range_other_on_cfg_norm')
class RangeNoiseRescaler(BaseNoiseRescaler):
    def __init__(self, noise_rescale_setup):
        super().__init__(noise_rescale_setup)
        assert len(noise_rescale_setup) == 2, 'incorrect len of noise_rescale_setup'
        self.lower_bound, self.upper_bound = noise_rescale_setup
        
    def _rescale(self, noises, index):
        cfg_noise_norm = torch.norm(noises['cfg']).item()
        other_noise_norm = torch.norm(noises['other']).item()
        
        ratio = other_noise_norm / cfg_noise_norm if cfg_noise_norm != 0 else 1.
        ratio_clipped = np.clip(ratio, self.lower_bound, self.upper_bound)
        if other_noise_norm != 0.:
            other_scale = ratio_clipped / ratio
        else:
            other_scale = 1.
        
        answer = {
            'cfg': 1.,
            'uncond': 1.,
            'other': other_scale 
        }
        return answer
    