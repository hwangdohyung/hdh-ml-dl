import numpy as np
import os
import random
from kerc_baseline import KERC22Baseline
import torch
from config_helper import  read_config_file

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything(SEED)
    conf = read_config_file(f'conf/config.ini')
    cfg = conf['train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = KERC22Baseline(device, cfg)
    baseline.train()


if __name__ == '__main__':
    main()
    
    