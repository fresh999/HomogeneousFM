import torch

import hydra
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os

from data_utils import sl2_from_lie_algebra_params, sl2_project, sl2_noise, sl2_to_lie_algebra_params, so3_from_lie_algebra_params, so3_noise, so3_project, so3_to_lie_algebra_params, stereo_project
from model import MLP
from solver.ode_solver import ODESolver
from utils.group_utils import SL2R, SO3R
from utils.model_wrapper import ModelWrapper



@hydra.main(version_base=None, config_path='../configs', config_name='config')
def visualize(cfg: DictConfig) -> None:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mode = cfg.mode
    if mode not in cfg.mlp:
        raise ValueError(f'Unknown mode: {mode}')

    model_cfg = cfg.mlp.reduced[mode]
    model = MLP(**model_cfg).to(device)

    if mode == 'sl2':
        G = SL2R()
    elif mode == 'so3':
        G = SO3R()

    for cp_id in cfg.visual.checkpoint_id:
        # load model
        checkpoint_path = os.path.join(
            cfg.visual.load_dir,
            f'reduced_lie_{mode}_checkpoint_iter_{cp_id}.pt'
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
        if mode == 'sl2':
            g_init = sl2_noise(cfg.visual.batch_size, device=device)
        elif mode == 'so3':
            g_init = so3_noise(cfg.visual.batch_size, device=device)

        x_init = G.log(g_init)
        if mode == 'sl2':
            p_init = sl2_to_lie_algebra_params(x_init)
        elif mode == 'so3':
            p_init = so3_to_lie_algebra_params(x_init)

        solver = ODESolver(velocity_model=vf)
        sol = solver.sample(
            time_grid=T,
            x_init=p_init,
            method='midpoint',
            step_size=step_size,
            return_intermediates=True
        )

        # project data to upper half-plane
        if mode == 'sl2':
            sol = sl2_from_lie_algebra_params(sol)
            sol = sl2_project(G.exp(sol))
        elif mode == 'so3':
            sol = so3_from_lie_algebra_params(sol)
            sol = stereo_project(so3_project(G.exp(sol)))

        sol = sol.cpu().numpy()
        T = T.cpu()

        fix, axs = plt.subplots(2, cfg.visual.n_steps // 2, figsize=(20, 20))
        axs = axs.flatten()

        for i in range(cfg.visual.n_steps):
            print(f'Processing time step {i}...')
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
                f'reduced_lie_{mode}_cp_{cp_id}.pdf'
            )
        )


if __name__ == '__main__':

    visualize()










