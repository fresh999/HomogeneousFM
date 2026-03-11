import torch

import hydra
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os

from data_utils import project, sl2_noise
from model import MLP
from solver.ode_solver import ODESolver
from utils.model_wrapper import ModelWrapper



@hydra.main(version_base=None, config_path='../configs', config_name='config')
def visualize(cfg: DictConfig) -> None:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP(
        input_features=cfg.mlp.input_features,
        output_features=cfg.mlp.output_features,
        width=cfg.mlp.width,
        depth=cfg.mlp.depth,
        activation=cfg.mlp.activation
    ).to(device)

    # load model
    checkpoint_path = os.path.join(
        cfg.visual.load_dir,
        f'checkpoint_iter_{cfg.visual.checkpoint_id}.pt'
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    vf = ModelWrapper(model)

    step_size = 0.05
    norm = cm.colors.Normalize(vmax=50, vmin=0)

    T = torch.linspace(0, 1, cfg.visual.n_steps, device=device)
    # sample noise from Sp(2) = SL(2, R)
    x_init = sl2_noise(cfg.visual.batch_size, device=device)

    solver = ODESolver(velocity_model=vf)
    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method='midpoint',
        step_size=step_size,
        return_intermediates=True
    )

    # project data to upper half-plane
    sol = project(sol)
    sol = sol.cpu().numpy()
    T = T.cpu()

    fix, axs = plt.subplots(2, cfg.visual.n_steps // 2, figsize=(20, 20))
    axs = axs.flatten()

    for i in range(cfg.visual.n_steps):
        print(f'Processing step {i}...')
        H, _, _, im = axs[i].hist2d(sol[i, :, 0], sol[i, :, 1], 300, range=((-5, 5), (-5, 10)), rasterized=True)

        cmin = 0.0
        cmax = np.quantile(H, 0.99)

        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
        im.set_norm(norm)

        axs[i].set_aspect('equal')
        axs[i].axis('off')
        axs[i].set_title('t= %.2f' % (T[i]))

    print('Creating pdf...')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            cfg.visual.output_dir,
            f'cp_{cfg.visual.checkpoint_id}.pdf'
        )
    )


if __name__ == '__main__':

    visualize()










