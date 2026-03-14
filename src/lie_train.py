import torch

from model import MLP
from data_utils import inf_train_gen, sl2_section, sl2_noise
from path.affine import CondOTProbPath
from utils.group_utils import SL2R

import hydra
from omegaconf import DictConfig
import os
import time
import warnings



@hydra.main(version_base=None, config_path='../configs', config_name='config')
def train(cfg: DictConfig) -> None:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}.')

    torch.manual_seed(cfg.seed)

    vf = MLP(
        input_features=cfg.mlp.input_features,
        output_features=cfg.mlp.output_features,
        width=cfg.mlp.width,
        depth=cfg.mlp.depth,
        activation=cfg.mlp.activation
    ).to(device)

    path = CondOTProbPath()

    optim = torch.optim.AdamW(vf.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    sl2 = SL2R()

    # training loop
    start_time = time.time()
    for i in range(cfg.training.iterations):
        optim.zero_grad()

        g_1 = sl2_section(inf_train_gen(batch_size=cfg.training.batch_size, device=device))
        g_0 = sl2_noise(batch_size=cfg.training.batch_size, device=device)
        t = torch.rand(g_1.shape[0]).to(device)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            x_1 = sl2.log(g_1).to(dtype=torch.float32)
            x_0 = sl2.log(g_0).to(dtype=torch.float32)

        path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

        # MSE loss
        loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()

        loss.backward()
        optim.step()

        if (i+1) % cfg.training.log_every == 0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} '
              .format(i+1, elapsed * 1000 / cfg.training.log_every, loss.item()))
            start_time = time.time()

            torch.save({
                'iteration': i,
                'model_state_dict': vf.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
            }, os.path.join(cfg.training.output_dir, f'checkpoint_iter_{i}.pt'))


if __name__ == '__main__':

    train()
