import torch

from model import MLP
from data_utils import inf_train_gen, sl2_section, sl2_noise, so3_section, so3_noise, stereo_inverse
from path.affine import CondOTProbPath

import hydra
from omegaconf import DictConfig
import os
import time



@hydra.main(version_base=None, config_path='../configs', config_name='config')
def train(cfg: DictConfig) -> None:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}.')

    torch.manual_seed(cfg.seed)

    mode = cfg.mode
    if mode not in cfg.mlp:
        raise ValueError(f'Unknown mode: {mode}.')

    model_cfg = cfg.mlp[mode]
    vf = MLP(**model_cfg).to(device)

    '''
    vf = MLP(
        input_features=cfg.mlp.input_features,
        output_features=cfg.mlp.output_features,
        width=cfg.mlp.width,
        depth=cfg.mlp.depth,
        activation=cfg.mlp.activation
    ).to(device)
    '''

    path = CondOTProbPath()

    optim = torch.optim.AdamW(vf.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # training loop
    start_time = time.time()
    for i in range(cfg.training.iterations):
        optim.zero_grad()

        if mode == 'sl2':
            x_1 = sl2_section(inf_train_gen(batch_size=cfg.training.batch_size, device=device, upper=True))
            x_0 = sl2_noise(batch_size=cfg.training.batch_size, device=device)
        elif mode == 'so3':
            data = inf_train_gen(batch_size=cfg.training.batch_size, device=device, upper=False)
            data = stereo_inverse(data)
            x_1 = so3_section(data)
            x_0 = so3_noise(batch_size=cfg.training.batch_size, device=device)
        else:
            raise ValueError('Unknown mode: {mode}.')

        t = torch.rand(x_1.shape[0]).to(device)

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
            }, os.path.join(cfg.training.output_dir, f'{mode}_checkpoint_iter_{i}.pt'))


if __name__ == '__main__':

    train()
