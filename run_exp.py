import json
import time
import click
import torch
import omegaconf

from PIL import Image
from omegaconf import OmegaConf
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm

from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry
from diffusion_core.utils import ClassRegistry, load_512
from diffusion_core.guiders.guidance_editing import GuidanceEditing
from diffusion_core.schedulers.sample_schedulers import DDIMScheduler
from diffusion_core.editing.p2p_pipe import P2PPipeBase
from diffusion_core.editing.masactrl_pipe import MasaCtrlEditPipe
from diffusion_core.utils import use_deterministic


pipelines = ClassRegistry()
pipelines.add_to_registry('ours')(GuidanceEditing)
pipelines.add_to_registry('p2p_base')(P2PPipeBase)
pipelines.add_to_registry('masactrl')(MasaCtrlEditPipe)


class DataRoot:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        
    def create_exp_root(self, config_path, root_name=None):
        ts = time.time()
        timestamp = int(round(ts))
        
        if root_name is None:
            root_name = f"{Path(config_path).stem}_{timestamp}"
        else:
            assert type(root_name) is str
        
        exp_root = self.root / root_name
        if exp_root.exists():
            print('[ WARNING ] exp root exists')
        else:
            print('[ INFO ] exp root is created: ', str(exp_root))
        
        exp_root.mkdir(parents=True, exist_ok=True)
        return exp_root
        
        
class BaseInferencer:
    def __init__(self, config):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.pipeline = self._get_pipeline(config)
        
    def __call__(self, im_path, src_prompt, trg_prompt, **kwargs):
        im = self._read_image(im_path)
        res = self.pipeline(im, src_prompt, trg_prompt, **kwargs)
        return res
    
    def _read_image(self, im_path):
        return load_512(im_path)
        
    def _get_scheduler(self, config):
        if config.scheduler_type not in diffusion_schedulers_registry:
            raise ValueError(f"Incorrect scheduler type: {config.scheduler_type}, possible are {diffusion_schedulers_registry}")
        scheduler = diffusion_schedulers_registry[config.scheduler_type]()
        return scheduler
    
    def _get_pipeline(self, config):
        scheduler = self._get_scheduler(config)
        model = diffusion_models_registry[config.model_name](scheduler)
        model.to(self.device)
        pipeline = pipelines[config.pipeline_type](model, config)
        return pipeline
    

def read_json(filename):
    with open(filename) as f:
        return json.load(f)
    
    
def get_test_data(ids_path, config):
    ids_data = read_json(ids_path)
    result_data = []
    for object_id, object_data in ids_data['id_data'].items():
    
        edit_type, im_path, src_prompt, trg_prompt = object_data
        
        if edit_type not in config.edit_types:
            continue
            
        result_data.append(
            (edit_type, (im_path, src_prompt, trg_prompt), object_id)
        )
        
    return result_data, ids_data['meta_info']['images_path']
    
    
def run_experiment(data_root, test_data, inferencer, images_root):
    pbar = tqdm(test_data, desc=f'Running {Path(data_root).name}')
    for edit_type, (im_path, src_prompt, trg_prompt), object_id in pbar:
        type_root = data_root / edit_type
        type_root.mkdir(exist_ok=True)
        im_path = f"{images_root}/{im_path}"
        save_path = type_root / f"{object_id}{Path(im_path).suffix}"
        if save_path.exists():
            continue

        res = inferencer(im_path, src_prompt, trg_prompt)
        Image.fromarray(res).save(save_path)

    
@click.command()
@click.option('--config_path', nargs=1, type=str)
@click.option('--output_root', nargs=1, type=str, default='res_data')
@click.option('--ids_path', nargs=1, type=str, default='testing/test_setup/ids_v5.json')
@click.option('--exp_root', nargs=1, type=str, default=None)
@click.option('--device', nargs=1, type=str, default='cuda:0')
def main(config_path, output_root, ids_path, exp_root, device):
    use_deterministic()
    print('Config path =', config_path)
    print('Output root =', output_root)
    config = OmegaConf.load(config_path)
    test_data, images_root = get_test_data(ids_path, config)
    
    root = DataRoot(output_root)
    data_root = root.create_exp_root(config_path, exp_root)
    OmegaConf.save(config, f"{data_root}/config.yaml")
    inferencer = BaseInferencer(config)
    run_experiment(data_root, test_data, inferencer, images_root)
    
    

if __name__ == '__main__':
    main()
