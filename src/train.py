import torch

from model import MLP
from data_utils import augment_data_random, inf_train_gen, section
from path.affine import CondOTProbPath

import os
import time


if __name__ == '__main__':

    train_cfg = {'lr': 0.001,
                 'batch_size': 4096,
                 'iterations': 10,
                 'print_every': 1,
                 'input_features': 4,
                 'output_features': 4,
                 'width': 512,
                 'depth': 4,
                 'activation': 'relu',
                 'n_samples': 16,
                 'bin': 'bin'
                 }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}.')

    torch.manual_seed(6)

    vf = MLP(
        input_features=train_cfg['input_features'],
        output_features=train_cfg['output_features'],
        width=train_cfg['width'],
        depth=train_cfg['depth'],
        activation=train_cfg['activation']
    ).to(device)

    path = CondOTProbPath()

    optim = torch.optim.Adam(vf.parameters(), lr=train_cfg['lr'])

    # training loop
    start_time = time.time()
    for i in range(train_cfg['iterations']):
        optim.zero_grad()

        x_1 = section(inf_train_gen(batch_size=train_cfg['batch_size'], device=device))
        x_0 = torch.randn_like(x_1)

        x_1 = augment_data_random(x_1, train_cfg['n_samples'])
        x_0 = augment_data_random(x_0, train_cfg['n_samples'])
        t = torch.rand(x_1.shape[0]).to(device)

        path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

        # MSE loss
        loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()

        loss.backward()
        optim.step()

        if (i+1) % train_cfg['print_every'] == 0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} '
              .format(i+1, elapsed * 1000 / train_cfg['print_every'], loss.item()))
            start_time = time.time()

            torch.save({
                'iteration': i,
                'model_state_dict': vf.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
            }, os.path.join(train_cfg['bin'], f'checkpoint_iter_{i}.pt'))




