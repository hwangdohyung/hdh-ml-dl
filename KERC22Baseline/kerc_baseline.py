import os.path

import torch
from torch.utils.data import DataLoader
from kd_emor_dataset import KDEmorPreprocessedDataset
from models import KERC22BaselineModel
import torch.optim as optim
import copy
from tqdm import tqdm
from loss import FocalLoss

'''
NOTE.
This baseline code is only for reference. The validation part has been omitted from the code. 
The baseline was trained in the presence of public test data, which we use in the first round of competition 
and the labels are not provided to the participants of KERC'22. 
Therefore, the baseline code only contains the training part...
Please split training set and use for validation.

'''

class KERC22Baseline():
    '''
    Baseline Trainer
    '''

    def __init__(self, device, config):
        super(KERC22Baseline, self).__init__()
        self.device = device
        self.conf = config
        self.data_dir = config['data_dir']
        self.log_dir = 'logs'
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)


    def train(self):
        # Data Loaders
        train_dataset = KDEmorPreprocessedDataset(self.data_dir, data_type='train')
        train_loader = DataLoader(train_dataset, self.conf['batch_size'], shuffle=True)

        # model
        model = KERC22BaselineModel(self.conf).to(self.device)
        optimizer = optim.Adam(model.parameters(), self.conf['learning_rate'])
        criterion = FocalLoss()
        NUM_EPOCHS = self.conf['epochs']
        best_loss = 10000

        # intital model state
        best_model_wts = copy.deepcopy(model.state_dict())
        epoch_tqdm = tqdm(total=NUM_EPOCHS, desc='Epoch', position=0)
        training_info = tqdm(total=0, position=1, bar_format='{desc}')
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0

            for sentence, scene_desc, target_speaker_ctx, other_speaker_ctx, scene_ctx, labels, _ in train_loader:
                scene_desc = scene_desc.to(self.device)
                sentence = sentence.to(self.device)
                target_speaker_ctx = target_speaker_ctx.to(self.device)
                other_speaker_ctx = other_speaker_ctx.to(self.device)
                scene_ctx = scene_ctx.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs  = model(sentence, scene_desc, target_speaker_ctx, other_speaker_ctx, scene_ctx)
                    loss = criterion(outputs, labels)
                    loss = loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * sentence.size(0)

            total_count = len(train_dataset)
            epoch_loss = running_loss / total_count
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            training_info.set_description_str(
                f'Epoch {epoch + 1}/{NUM_EPOCHS},  Loss:{epoch_loss:.4f}')
            epoch_tqdm.update(1)

        # load best model weights and save
        model.load_state_dict(best_model_wts)
        torch.save(model, f'{self.log_dir}/saved_model.pt')

