import torch
from torch import nn
import math
import torch.nn.functional as F

# Scaled dot product attention
class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, q, k, v):
        # q,k,v ---- [B, T, X]
        k = k.permute(0, 2, 1) # [B, T, K] -> [B, K, T]
        scaling_factor = 1.0 / math.sqrt(q.size(2))
        e = torch.bmm(q, k) # [B, T, Q] * [B, T, V] -> [B, T, T]
        attention_weight = torch.nn.functional.softmax(e.mul_(scaling_factor), dim=2) # softmax taken across time axis
        return attention_weight.matmul(v)

# Baseline Model
class KERC22BaselineModel(nn.Module):
    def __init__(self, model_params):
        super(KERC22BaselineModel, self).__init__()
        self.att_speaker_self = Attention()
        self.x_att_speakers = Attention()
        self.x_att_desc_on_scene = Attention()
        self.x_att_scene_on_sent = Attention()
        self.self_att_fused = Attention()

        self.shared_linear = nn.Linear(768, 512)
        fusion_dim = 3840

        self.clf = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, model_params['clf_out']),  # clf_out = 3
            nn.Sigmoid())

    def forward(self, sentence, target_spkr_snts, other_spkr_snts, scene_desc, scene_sent):
        sentence = sentence.view(-1, sentence.shape[1], 1)
        scene_desc = scene_desc.view(-1, sentence.shape[1], 1)
        target_spkr_snts = target_spkr_snts.view(-1, sentence.shape[1], 1)
        other_spkr_snts = other_spkr_snts.view(-1, sentence.shape[1], 1)
        scene_sent = scene_sent.view(-1, sentence.shape[1], 1)

        q21_, k21_, v21_ = other_spkr_snts.clone(), target_spkr_snts.clone(), target_spkr_snts.clone()
        speaker_context = self.x_att_speakers(q21_, k21_, v21_)

        q1_, k1_, v1_ = scene_sent.clone(), scene_sent.clone(), scene_desc.clone()
        scene_context = self.x_att_desc_on_scene(q1_, k1_, v1_)

        sentence_ = sentence.view(sentence.size(0), -1)
        scene_context_ = scene_context.view(scene_context.size(0), -1)
        speaker_context_ = speaker_context.view(speaker_context.size(0), -1)

        sentence_shared = self.shared_linear(sentence_)
        scene_context_shared = self.shared_linear(scene_context_)
        speaker_context_shared = self.shared_linear(speaker_context_)
        fused_intermediate = torch.cat([sentence, scene_context, speaker_context], 1)

        q3_, k3_, v3_ = fused_intermediate.clone(), fused_intermediate.clone(), fused_intermediate.clone()
        z_self_att = self.self_att_fused(q3_, k3_, v3_)

        final_ftrs = z_self_att.view(z_self_att.size(0), -1)
        final_ftrs = torch.cat([final_ftrs, sentence_shared, scene_context_shared, speaker_context_shared], 1)

        x = self.clf(final_ftrs)

        return x