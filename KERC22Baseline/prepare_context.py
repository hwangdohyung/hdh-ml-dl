import os
import pandas as pd
from tqdm import tqdm
from config_helper import read_config_file


class ConversationContext():
    """
    Get speaker level and scene level contexts
    """
    def __init__(self, data_dir, data_type='train'):
        self.data_type = data_type
        self.data_dir = data_dir
        self.temp_dir = f'{data_dir}/tmp'
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    # Speaker context is expressed in terms of:
    # Target Speaker Context: represented by the sentences spoken by the speaker of the target sentence.
    # Other Speaker Context: represented by other sentences except those spoken by the speaker of the target sentence.
    def __get_speaker_context(self, data_df, scene_groups, sample_idx):
        sample_row =  data_df.loc[sample_idx]
        start_idx = scene_groups[sample_row['scene']].min() #first sample in a scene
        target_spkr_  = ""
        other_spkr_ = ""
        for idxx in range(start_idx, sample_idx): #only past samples
            if data_df.loc[sample_idx]['person'] in data_df.loc[idxx]['person']:
                target_spkr_ = target_spkr_ + " " + data_df.loc[idxx]['sentence']
            else:
                other_spkr_ = other_spkr_ + " " +  data_df.loc[idxx]['sentence']

        # when there are no sentences add empty space
        if len(target_spkr_) == 0:
            target_spkr_ = " "
        if len(other_spkr_) == 0:
            other_spkr_ = " "
        return target_spkr_, other_spkr_


    # Scene context represents the sequential information flow in the scene,
    # and is represented by conversation history in the scene.
    def __get_scene_context(self, data_df, scene_groups, sample_idx):
        sample_row =  data_df.loc[sample_idx]
        start_idx = scene_groups[sample_row['scene']].min() #first sample in a scene
        scene_sentences  = " "
        for idxx in range(start_idx, sample_idx): #only past samples
            scene_sentences  += data_df.loc[idxx]['sentence']
        return scene_sentences

    def extract_context(self):
        print(f'Preprocess {self.data_type} data..')
        try:
            data_df = pd.read_csv(f'{self.data_dir}/{self.data_type}_data.tsv', delimiter='\t')
        except FileNotFoundError:
            print(f"{self.data_type} : File Not Found (Please check data files in {self.data_dir}.)")
            return
        sentence_ids = []
        target_sents = []
        other_sents = []
        scene_sents = []
        for idx in tqdm(range(len(data_df))):
            scene_groups = data_df.groupby(by='scene').indices
            target_, other_ = self.__get_speaker_context(data_df, scene_groups, idx)
            sentence_ids.append(data_df['sentence_id'][idx])
            target_sents.append(target_)
            other_sents.append(other_)
            scene_ = self.__get_scene_context(data_df, scene_groups, idx)
            scene_sents.append(scene_)
        data_df['sentence_id'] = sentence_ids
        data_df['target_speaker_ctx'] = target_sents
        data_df['other_speaker_ctx'] = other_sents
        data_df['scene_sents'] = scene_sents
        data_df['scene_sents'].mask(data_df['scene_sents'] == '', ' ', inplace=True)
        data_df.to_csv(f"{self.temp_dir}/df_{self.data_type}_extended.tsv", sep="\t", index=False)


def main():
    conf = read_config_file(f'conf/config.ini')
    cfg = conf['train']
    for data_type in ['train', 'public_test', 'private_test']:
        ctx = ConversationContext(cfg['data_dir'], data_type)
        ctx.extract_context()


if __name__ == "__main__":
    main()
    
    
    