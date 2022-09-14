import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# NOTE: Private and Public Test Labels will not be available to the Participants.
class KDEmorPreprocessedDataset(Dataset):
    def __init__(self, data_dir, data_type='train'):

        assert data_type in ['train', 'public_test', 'private_test'], f'Unknown data_type {data_type}'

        self.data_type = data_type
        self.labels = ['dysphoria', 'neutral', 'euphoria']

        self.data_df = pd.read_csv(f'{data_dir}/{self.data_type}_data.tsv', delimiter='\t')

        self.sentence = torch.load(f'{data_dir}/preprocessed/{data_type}_sentence_bert_features.pt')
        self.scene_desc = torch.load(f'{data_dir}/preprocessed/{data_type}_context_bert_features.pt')
        self.target_speaker_ctx = torch.load(f'{data_dir}/preprocessed/{data_type}_target_speaker_ctx_bert_features.pt')
        self.other_speaker_ctx = torch.load(f'{data_dir}/preprocessed/{data_type}_other_speaker_ctx_bert_features.pt')
        self.scene_ctx = torch.load(f'{data_dir}/preprocessed/{data_type}_scene_sents_bert_features.pt')

        if data_type not in ['public_test', 'private_test']: # Labels are not available for test data
            self.label_df = pd.read_csv(f'{data_dir}/{data_type}_labels.csv')

        self.len = self.data_df.shape[0]

    def __getitem__(self, index):
        sentence_id = self.data_df['sentence_id'][index]

        label = np.NaN # Not used as public_test and private_test data are not available.
        if self.data_type not in ['public_test', 'private_test']:  # Labels are not available for test data
            label = self.label_df[self.label_df['sentence_id'] == sentence_id]['label'].values[0]
            label = self.labels.index(label)

        sentence = self.sentence[index]
        scene_desc = self.scene_desc[index]
        target_speaker_ctx = self.target_speaker_ctx[index]
        other_speaker_ctx = self.other_speaker_ctx[index]
        scene_ctx = self.scene_ctx[index]
        return sentence, scene_desc, target_speaker_ctx, other_speaker_ctx, scene_ctx, label, sentence_id

    def __len__(self):
        return self.len