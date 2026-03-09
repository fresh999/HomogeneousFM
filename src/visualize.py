import torch

from matplotlib import cm

from model import MLP


if __name__ == '__main__':

    visualize_cfg = {'batch_size': 2 ** 14,
                     'n_steps': 10,
                     'bin': 'bin'}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
