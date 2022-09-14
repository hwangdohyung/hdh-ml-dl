from config_helper import read_config_file
from kd_emor_dataset import KDEmorPreprocessedDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import os
import random
import warnings

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def eval(cfg, device, data_type, model):
    '''
    cfg
    device
    data_type : train, public_test, private_test
    log_dir
    '''
    test_dataset = KDEmorPreprocessedDataset(data_dir=cfg['data_dir'], data_type=data_type)
    test_loader = DataLoader(test_dataset, cfg['batch_size'], shuffle=True)

    model.eval()

    sample_ids = []
    pred_labels = []

    with torch.no_grad():
        for sentence, scene_desc, target_speaker_ctx, other_speaker_ctx, scene_ctx, _, sample_id in test_loader:
            # move ot gpu
            scene_desc = scene_desc.to(device)
            sentence = sentence.to(device)
            target_speaker_ctx = target_speaker_ctx.to(device)
            other_speaker_ctx = other_speaker_ctx.to(device)
            scene_ctx = scene_ctx.to(device)

            outputs = model(sentence, scene_desc, target_speaker_ctx, other_speaker_ctx, scene_ctx)

            sample_ids.extend(sample_id.detach().cpu().numpy())
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            pred_labels.extend(preds)

    return sample_ids, pred_labels



def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = 'logs'
    conf = read_config_file(f'conf/config.ini')
    cfg = conf['train']


    ## CHANGE THIS
    eval_test = False

    sample_ids = []
    pred_labels = []
    test_sets = ['public_test', 'private_test'] if eval_test else ['public_test']
    for test_set in test_sets:
        model = torch.load(f'{log_dir}/saved_model.pt')
        ids, preds = eval(cfg, device, test_set, model)
        sample_ids.extend(ids)
        pred_labels.extend(preds)

    submission_df = pd.DataFrame()
    submission_df["Id"] = sample_ids
    submission_df["Predicted"] = [['dysphoria', 'neutral', 'euphoria'][x] for x in pred_labels]
    submission_df.to_csv(f'{log_dir}/submission_TEAM_NAME.csv', index=False)
    print(f"Saved '{log_dir}/submission_TEAM_NAME.csv")


if __name__ == '__main__':
    main()
