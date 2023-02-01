from PIL import Image
from cog import BasePredictor, Input, Path
import base64
from io import BytesIO
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import load_image
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf
import yaml
import tqdm
import torch
import numpy as np
import os

from saicinpainting.evaluation.utils import move_to_device


cache_dir = "./cache"
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Predictor(BasePredictor):
    def setup(self):
        train_config_path = os.path.join("./big-lama", 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        self.out_ext = ".png"
        checkpoint_path = os.path.join("./big-lama",
                                       'models',
                                       "best.ckpt")
        self.model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(device)

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        img_bytes: str = Input(default=""),
        mask_bytes: str = Input(default=""),
    ) -> dict:
        image = load_image(BytesIO(base64.b64decode(img_bytes)), mode='RGB')
        mask = load_image(BytesIO(base64.b64decode(mask_bytes)), mode='L')
        result = dict(image=image, mask=mask[None, ...])

        batch = default_collate([result])
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self.model(batch)
            cur_res = batch["inpainted"][0].permute(
                1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

        inpainted = Image.fromarray(cur_res)
        inpainted_path = f"/tmp/outdepth.png"
        inpainted.save(inpainted_path)
        return dict(inpainted=Path(inpainted_path))
