import json
import os

import torch
import torch.nn as nn
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # prompts[2,77,512]         真假prompt
        # tokenized_prompts[2,77]   tokenize
        x = prompts + self.positional_embedding.type(self.dtype)  # [2,77,512]+[77,512]==>[2,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND  # ==>[77,2,512]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  # ==>[2,77,512]
        x = self.ln_final(x).type(self.dtype)  # ==>[2,77,512]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 用[2]和[2]取出x[2,77,512]中的==>[2,512]
        # [2,512]@[512,512]==>[2,512]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x  # [2,512]


class PromptLearner(nn.Module):
    def __init__(self, args, clip_model, label_names):
        super().__init__()
        n_cls = 10  # real fake
        n_ctx = args.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        print("Initializing a generic context")  # 无初始化语句、不使用类特定contexts
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # 优化  # [16,512]

        class_names = label_names[:]
        class_names[9] = 'Thalamus transverse section, ' \
                         'showing the strong echo ring of the skull, the cerebral falx, ' \
                         'the cavity of septum pellucidum, the thalami on both sides, and the Sylvian fissure.'
        class_names[5] = 'Lateral ventricle transverse section, ' \
                         'showing the strong echo ring of the skull, ' \
                         'the cerebral falx, the cavity of septum pellucidum, ' \
                         'the anterior horn and posterior horn of the lateral ventricle, ' \
                         'and the choroid plexus within the ventricle.'

        prompts = [f'{prompt_prefix} {label}.' for label in class_names]  # 0真1假

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # [2,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # 不优化
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS  # [2,1,512]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS  # [2,60,512]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # [2,77]由XXX...real/fake初始化的tokenize
        # self.name_lens = name_lens
        self.class_token_position = args.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx  # [16,512]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # ==>[2,16,512]

        prefix = self.token_prefix  # [2,1,512]
        suffix = self.token_suffix  # [2,60,512]

        if self.class_token_position == "end":
            prompts = torch.cat(  # ([2,1,512],[2,16,512],[2,60,512])=cat=>[2,77,512]
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts  # [2,77,512]  分别为a和b的prompts


class CLIP_Model(nn.Module):
    def __init__(self, args):
        super(CLIP_Model, self).__init__()
        self.clip_model = clip.load(args.backbone, device)[0].float()

        with open(os.path.join(args.data_path, 'label_names.json')) as f:
            self.label_names = json.load(f)

        self.prompt_learner = PromptLearner(args, self.clip_model, self.label_names)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip_model)


        self.requires_grad_(False)
        self.clip_model.visual.requires_grad_(True)
        self.prompt_learner.requires_grad_(True)

    def forward(self, image, labels=None, clip_loss=True, cls_loss=False):
        logits_per_image_clip, logits_per_text_clip, logits_per_image_cls = None, None, None

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        prompts = self.prompt_learner()  # [2,77,512]  分别为a和b的prompts
        tokenized_prompts = self.tokenized_prompts  # [2,77]tokenize

        if clip_loss and self.training:
            # 根据标签排列
            prompts_list = []  # ==>bs*[77,512]
            tokenized_prompts_list = []
            for label in labels:
                prompts_list.append(prompts[label])
                tokenized_prompts_list.append(tokenized_prompts[label])
            prompts = torch.stack(prompts_list)
            tokenized_prompts = torch.stack(tokenized_prompts_list)
            text_features_clip = self.text_encoder(prompts, tokenized_prompts)

            text_features_clip = text_features_clip / text_features_clip.norm(dim=1, keepdim=True)
            logits_per_image_clip = logit_scale * image_features @ text_features_clip.t()
            logits_per_text_clip = logits_per_image_clip.t()
            
        if cls_loss or (not self.training):
            text_features_cls = self.text_encoder(prompts, tokenized_prompts)
            text_features_cls = text_features_cls / text_features_cls.norm(dim=1, keepdim=True)
            logits_per_image_cls = logit_scale * image_features @ text_features_cls.t()
    
        return logits_per_image_clip, logits_per_text_clip, logits_per_image_cls


