import json
import lpips
import clip
import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from diffusion_core.utils.exp_utils import show_code
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .aesthetic_utils import normalized


def read_json(filename):
    with open(filename) as f:
        return json.load(f)


class DiffusionTestingDataset(Dataset):
    def __init__(
        self,
        exp_name,
        images_root,
        ids_file
    ):
        super().__init__()
        self.exp_root = Path(exp_name)
        assert self.exp_root.exists(), 'No such experiment'
        
        self.images_root = Path(images_root).resolve()
        self.ids = read_json(ids_file)['id_data']        

        sorted_ids = sorted([i for i in self.ids], key=lambda x: int(str(x)))
        self.idx_to_id = dict(
            [(idx, id_) for idx, id_ in zip(range(len(sorted_ids)), sorted_ids)]
        )
        self.to_tensor = ToTensor()
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        object_id = self.idx_to_id[idx]
        src_im, trg_im = self._get_images(object_id)
        return object_id, src_im, trg_im
    
    def _process_image(self, image):
        tensor_im = self.to_tensor(image)
        return (tensor_im - 0.5) * 2
    
    def _get_images(self, object_id):
        edit_type, rel_im_path, _, _ = self.ids[object_id]
        
        src_im_path = self.images_root / rel_im_path
        trg_im_path = self.exp_root / edit_type / f"{object_id}{Path(rel_im_path).suffix}"
        
        if not trg_im_path.exists():
            return None, None
        
        src_im = Image.open(src_im_path)
        trg_im = Image.open(trg_im_path)
        return self._process_image(src_im), self._process_image(trg_im)
    
    
def load_clip(model_name: str, device: str):
    """
    Get clip model with preprocess which can process torch images in value range of [-1, 1]

    Parameters
    ----------
    model_name : str
        CLIP-encoder type

    device : str
        Device for clip-encoder

    Returns
    -------
    model : nn.Module
        torch model of downloaded clip-encoder

    preprocess : torchvision.transforms.transforms
        image preprocess for images from stylegan2 space to clip input image space
            - value range of [-1, 1] -> clip normalized space
            - resize to 224x224
    """
    
    model, preprocess = clip.load(model_name, device=device)
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
        *preprocess.transforms[:2],
        preprocess.transforms[-1]
    ])

    return model, preprocess


class ClipEncoder:
    def __init__(self, visual_encoder, device):
        self.model, self.preprocess = load_clip(visual_encoder, device)
        self.device = device
    
    def encode_text(self, text: str):
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image: Image.Image):
        image_features = self.model.encode_image(self.preprocess(image))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features


class BaseMetric:
    def __init__(
        self, 
        exp_name,
        ids_file,
        batch_size=8,
        device='cuda:0'
    ):
        self.exp_root = Path(exp_name)
        assert self.exp_root.exists(), 'No such experiment'
        
        json_test_data = read_json(ids_file)
        self.ids = json_test_data['id_data']
        self.images_root = json_test_data['meta_info']['images_path']
        
        self.dataset = DiffusionTestingDataset(
            exp_name,
            self.images_root,
            ids_file
        )
        self.loader = DataLoader(self.dataset, batch_size=batch_size)
        self.device = device
    
    @torch.no_grad()
    def __call__(self):
        metric_result = {}
        
        for ids, src_images, trg_images in tqdm(self.loader, desc=self.exp_root.stem, leave=True):
            
            metric_result.update(self.calc_object(
                ids, src_images.to(self.device), trg_images.to(self.device)
            ))
            
        return metric_result
    
    def calc_object(
        self, 
        ids,
        src_images,
        trg_images
    ):
    
        raise NotImplementedError('Implement self.calc_object()')


class LPIPSMetric(BaseMetric):
    def __init__(self, exp_name, ids_file, **kwargs):
        super().__init__(exp_name, ids_file, **kwargs)
        
        assert 'device' in kwargs, 'set device up'
        
        self.alex_lpips = lpips.LPIPS(net='alex').to(kwargs['device'])
        self.vgg_lpips = lpips.LPIPS(net='vgg').to(kwargs['device'])
        self.to_tensor = ToTensor()
        
    def calc_object(
        self, 
        ids,
        src_images,
        trg_images
    ):
        
        alex_scores = self.alex_lpips(trg_images, src_images)
        vgg_scores = self.vgg_lpips(trg_images, src_images)
        
        bs = alex_scores.size(0)
        alex_scores = alex_scores.squeeze()
        vgg_scores = vgg_scores.squeeze()
        if bs > 1:
            alex_scores = alex_scores.chunk(alex_scores.size(0))
            vgg_scores = vgg_scores.chunk(vgg_scores.size(0))
        else:
            alex_scores = [alex_scores]
            vgg_scores = [vgg_scores]
        
        alex_scores = {
            f"alex_lpips/{ob_id}": metric.item() for ob_id, metric in zip(ids, alex_scores)
        }
        
        vgg_scores = {
            f"vgg_lpips/{ob_id}": metric.item() for ob_id, metric in zip(ids, vgg_scores)
        }
        
        return {
            **alex_scores,
            **vgg_scores
        }

    
class CLIPMetric(BaseMetric):
    def __init__(self, exp_name, ids_file, **kwargs):
        assert 'device' in kwargs, 'set device up'
        
        super().__init__(exp_name, ids_file, **kwargs)
        
        self.vit_b_32 = ClipEncoder('ViT-B/32', kwargs['device'])
        self.vit_l_14 = ClipEncoder('ViT-L/14', kwargs['device'])
        self.to_tensor = ToTensor()
        self.sim = nn.CosineSimilarity()
    
    def calc_object(
        self, 
        ids,
        src_images,
        trg_images
    ):
        
        trg_prompts = [self.ids[k][3] for k in ids]
        bs = len(trg_prompts)
        
        b_32_im_features = self.vit_b_32.encode_image(trg_images)
        b_32_text_features = self.vit_b_32.encode_text(trg_prompts)
        b_32_scores = self.sim(b_32_im_features, b_32_text_features).squeeze()
        
        l_14_im_features = self.vit_l_14.encode_image(trg_images)
        l_14_text_features = self.vit_l_14.encode_text(trg_prompts)
        l_14_scores = self.sim(l_14_im_features, l_14_text_features).squeeze()
        
        if bs > 1:
            l_14_scores = l_14_scores.chunk(bs)
            b_32_scores = b_32_scores.chunk(bs)
        else:
            l_14_scores = [l_14_scores]
            b_32_scores = [b_32_scores]
        
        l_14_scores = {
            f"l_14_clip/{ob_id}": metric.item() for ob_id, metric in zip(ids, l_14_scores)
        }
        
        b_32_scores = {
            f"b_32_clip/{ob_id}": metric.item() for ob_id, metric in zip(ids, b_32_scores)
        }
        
        return {
            **l_14_scores,
            **b_32_scores
        }

    
class AestheticMetric(BaseMetric):
    def __init__(self, exp_name, ids_file, **kwargs):
        super().__init__(exp_name, ids_file, **kwargs)
        assert 'device' in kwargs, 'set device up'
        self.device = kwargs['device']
        model_path = kwargs['model_path'] if 'model_path' in kwargs else 'testing/weights/ava+logos-l14-linearMSE.pth'
        self.score_model = self._load_model_acu(768, model_path, kwargs['device'])
        self.clip_model = ClipEncoder("ViT-L/14", device=kwargs['device'])
        self.ir_model = self._load_model_ir()
        
    def _load_model_ir(self):
        import ImageReward as RM
        return RM.load("ImageReward-v1.0")
    
    def _load_model_acu(self, input_dim, model_path, device='cuda'):
        from .aesthetic_utils import ACUModel
        
        model = ACUModel(input_dim)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        return model
    
    def get_scores_acu(self, trg_images):
        image_features = self.clip_model.encode_image(trg_images)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.score_model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        prediction = prediction.chunk(prediction.size(0), dim=0)
        return prediction
    
    def get_single_score_ir(self, prompt, image):
        score = self.ir_model.score(
            prompt,
            image
        )
        return score
    
    def get_scores_ir(self, ids, trg_images):
        prompts = [self.ids[i][3] for i in ids]
        images = [transforms.ToPILImage()((im + 1)/2) for im in trg_images]
        scores = [self.get_single_score_ir(p, im) for p, im in zip(prompts, images)]
        return scores
        
    def calc_object(
        self,
        ids,
        src_images,
        trg_images
    ):

        acu_scores = self.get_scores_acu(trg_images)
        acu_scores = {
            f"aesthetic_acu/{ob_id}": metric.item() for ob_id, metric in zip(ids, acu_scores)
        }
        
        ir_scores = self.get_scores_ir(ids, trg_images)
        ir_scores = {
            f"aesthetic_ir/{ob_id}": metric for ob_id, metric in zip(ids, ir_scores)
        }
                
        return {
            **acu_scores,
            **ir_scores
        }
