import os
import shutil
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import warnings
from config_helper import read_config_file
warnings.filterwarnings("ignore")


class BERTFeatures():
    """
    Preprocessing train, public_test, and private_test data to generate the speaker-level context
    """
    def __init__(self, data_dir, data_type='train'):
        self.data_dir = data_dir
        self.output_dir = f"{self.data_dir}/preprocessed"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_type = data_type

        try:
            self.file_name = f'{self.data_dir}/tmp/df_{data_type}_extended.tsv'
            self.data_df = pd.read_csv(self.file_name, delimiter='\t')
            self.data_df['context'].fillna(' ', inplace=True)
            self.SENTENCE_MAX_LEN = 200
            self.CONTEXT_MAX_LEN = 200
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt', do_lower_case=False)
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased").to(self.device)
            self.file_exists = True
        except FileNotFoundError:
            self.file_exists = False


    def __get_bert_features(self, text_list):
        tokenized_ = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        tokenized_on_gpu = {k: torch.tensor(v).to(self.device) for k, v in tokenized_.items()}
        with torch.no_grad():
            hidden_ = self.model(**tokenized_on_gpu)  # dim : [batch_size(nr_sentences), tokens, emb_dim]
            cls_tokens = hidden_.last_hidden_state[:, 0, :]
        return cls_tokens

    def run(self):
        if not self.file_exists:
            print(f"{self.data_type} : File Not Found:: (Please check if {self.file_name} exists.) Run prepare_context.py first")
            return
        # process in batches of 16 samples, process in parts train: 7224 public_test: 2759 private_test: 2306
        dict_parts = {"train": (459, 16, 7339), "public_test": (161, 16, 2566), "private_test": (149, 16, 2384)}
        a, b, total = dict_parts[self.data_type]
        for modality in ['sentence', 'context', 'target_speaker_ctx', 'other_speaker_ctx', 'scene_sents']:
            print(f"Preprocess {self.data_type}  - {modality} features")
            temp_dir_ = f'{self.output_dir}/temp_'
            if not os.path.exists(temp_dir_):
                os.mkdir(temp_dir_)
            for i in tqdm(range(a)):
                start = (i * b) + 1
                end = (i * b) + b
                if end > total:
                    end = total
                partial_datadf = self.data_df[start - 1:end]
                text_list = partial_datadf[modality].values.tolist()
                # saving features to file because of out of memory error, for device with memory more than 10GB it could be stacked in memory and saved at once.
                torch.save(self.__get_bert_features(text_list), f'{temp_dir_}/{i}_{self.data_type}_{modality}_bert_features.pt')

            tensor_modality = torch.load(f'{temp_dir_}/0_{self.data_type}_{modality}_bert_features.pt')
            for i in range(1, a):
                tensor_modality = torch.vstack([tensor_modality, torch.load(f'{temp_dir_}/{i}_{self.data_type}_{modality}_bert_features.pt')])
            torch.save(tensor_modality, f'{self.output_dir}/{self.data_type}_{modality}_bert_features.pt')
            shutil.rmtree(temp_dir_, ignore_errors=True)



def main():
    conf = read_config_file(f'conf/config.ini')
    cfg = conf['train']
    for data_type in ['train', 'public_test', 'private_test']:
        bert_extractor = BERTFeatures(cfg['data_dir'], data_type)
        bert_extractor.run()


if __name__ == "__main__":
    main()
    
    
    
    