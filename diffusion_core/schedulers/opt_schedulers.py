import numpy as np
from diffusion_core.utils.class_registry import ClassRegistry


opt_registry = ClassRegistry()


@opt_registry.add_to_registry('constant')
class OptScheduler:
    def __init__(self, max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop):
        self.max_inner_steps = max_inner_steps
        self.max_ddim_steps = max_ddim_steps
        self.early_stop_epsilon = early_stop_epsilon
        self.plateau_prop = plateau_prop
        self.inner_steps_list = np.full(self.max_ddim_steps, self.max_inner_steps)

    def __call__(self, ddim_step, inner_step, loss=None):
        return inner_step + 1 >= self.inner_steps_list[ddim_step]
    
    def name(self):
        return 'constant'
    

@opt_registry.add_to_registry('loss')
class LossOptScheduler(OptScheduler):
    def __call__(self, ddim_step, inner_step, loss):
        if loss < self.early_stop_epsilon + ddim_step * 2e-5:
            return True
        self.inner_steps_list[ddim_step] = inner_step + 1
        return False
    
    def name(self):
        return 'by loss'
    

@opt_registry.add_to_registry('sin')
class SinOptScheduler(OptScheduler):    
    def __init__(self, max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop):
        super().__init__(max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop)
        
        timesteps = np.arange(self.max_ddim_steps, 0, -1)
        self.inner_steps_list = np.ceil(np.sin(np.pi * timesteps / self.max_ddim_steps) * self.max_inner_steps).astype(np.int64)
        
    def name(self):
        return 'sin'
    

@opt_registry.add_to_registry('log trapez')
class LogTrapezOptScheduler(OptScheduler):    
    def __init__(self, max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop):
        super().__init__(max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop)
        
        timesteps = np.arange(self.max_ddim_steps, 0, -1)
        l = int(self.max_ddim_steps / 2 - self.plateau_prop * self.max_ddim_steps / 2)
        r = int(self.max_ddim_steps / 2 + self.plateau_prop * self.max_ddim_steps / 2)
        left = np.ceil(1 + np.log(self.max_ddim_steps - timesteps[:l] + 1) / np.log(self.max_ddim_steps + 1) * self.max_inner_steps).astype(np.int64)
        middle = np.full(r - l, self.max_inner_steps)
        right = np.ceil(1 + np.log(timesteps[r:]) / np.log(self.max_ddim_steps + 1) * self.max_inner_steps).astype(np.int64)
        
        self.inner_steps_list = np.concatenate((left, middle, right))

    def name(self):
        return 'log trapezoid'
    
    
@opt_registry.add_to_registry('geom trapez')
class GeomTrapezOptScheduler(OptScheduler):    
    def __init__(self, max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop):
        super().__init__(max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop)
        
        l = int(self.max_ddim_steps / 2 - self.plateau_prop * self.max_ddim_steps / 2)
        r = int(self.max_ddim_steps / 2 + self.plateau_prop * self.max_ddim_steps / 2)
        left = np.ceil(np.geomspace(1, self.max_inner_steps, num=self.max_ddim_steps - r)).astype(np.int64)
        middle = np.full(r - l, self.max_inner_steps)
        right = np.ceil(np.geomspace(self.max_inner_steps, 1, num=l)).astype(np.int64)
        
        self.inner_steps_list = np.concatenate((left, middle, right))

    def name(self):
        return 'geom trapezoid'
