import json
import os.path

import torch
import torch.nn as nn
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIP_Model(nn.Module):
    def __init__(self, args):
        super(CLIP_Model, self).__init__()
        self.clip_model = clip.load(args.backbone, device)[0].float()
        with open(os.path.join(args.data_path, 'label_names.json')) as f:
            self.label_names = json.load(f)

        self.requires_grad_(True)
        self.clip_model.requires_grad_(False)
        self.clip_model.visual.requires_grad_(True)

    def forward(self, image, labels=None, clip_loss=True, cls_loss=False):
        logits_per_image_clip, logits_per_text_clip, logits_per_image_cls = None, None, None

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_str = [f'An image of a {name}' for name in self.label_names]

        if clip_loss and self.training:
            text_list = [text_str[label] for label in labels]
            text = clip.tokenize(text_list).to(device)
            text_features_clip = self.clip_model.encode_text(text)
            text_features_clip = text_features_clip / text_features_clip.norm(dim=1, keepdim=True)
            logits_per_image_clip = logit_scale * image_features @ text_features_clip.t()
            logits_per_text_clip = logits_per_image_clip.t()
        if cls_loss or (not self.training):
            text_list = text_str
            text = clip.tokenize(text_list).to(device)
            text_features_cls = self.clip_model.encode_text(text)
            text_features_cls = text_features_cls / text_features_cls.norm(dim=1, keepdim=True)
            logits_per_image_cls = logit_scale * image_features @ text_features_cls.t()
    
        return logits_per_image_clip, logits_per_text_clip, logits_per_image_cls