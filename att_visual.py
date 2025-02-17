import os.path
import sys
sys.path.append(r'.\Transformer-MM-Explainability')
import CLIP.clip as clip
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization

import argparse

parser = argparse.ArgumentParser(description='PyTorch clip')
# 训练参数
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--workers', default=0, type=int, help='workers')
# 数据参数
parser.add_argument('--image_path', default=r"C:\Users\L\Desktop\code_clip\data_ten\0\丘大妹_001_132036.jpg", type=str)
parser.add_argument('--save_path', default=r'./test_dir', type=str)
parser.add_argument('--data_path', default=r'C:\Users\L\Desktop\code_clip\data_ten', type=str)
parser.add_argument('--model_path', default='ckptsave/final/ckpt_best.pth', type=str)
# 模型参数
parser.add_argument('--backbone', default='ViT-B/16', type=str)
# coop相关参数
parser.add_argument('--N_CTX', default=12, type=int, help='number of context vectors')
parser.add_argument('--CLASS_TOKEN_POSITION', default='end', type=str, help='middle or end or front')

args = parser.parse_args()


start_layer =  -1#@param {type:"number"}

start_layer_text =  -1#@param {type:"number"}

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    # batch_size = texts.shape[0]
    batch_size = 1
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
      # calculate index of last layer
      start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]
    return image_relevance


def save_image_relevance(image_relevance, image, orig_image, orig_path='original_image.png',
                         heatmap_path='heatmap_image.png'):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    # Process image_relevance
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    # Process original image
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    # Create heatmap visualization
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    # Save original image
    plt.figure()
    plt.imshow(orig_image)
    plt.axis('off')
    plt.savefig(orig_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Save heatmap image
    plt.figure()
    plt.imshow(vis)
    plt.axis('off')
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
model.eval()

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# 图像处理
image = Image.open(args.image_path).convert('RGB')
width, height = image.size
scale = 224 / min(width, height)
new_width = int(width * scale)
new_height = int(height * scale)
image = image.resize((new_width, new_height))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.CenterCrop(224)
])
image = transform(image)
img = image.unsqueeze(0).to(device)

texts = ["abdominal circumference transverse section"]
text = clip.tokenize(texts).to(device)

R_image = interpret(model=model, image=img, texts=text, device=device)
batch_size = text.shape[0]

save_image_relevance(R_image[0], img, orig_image=Image.open(args.image_path),
                     orig_path=os.path.join(args.save_path, 'att_original_image.png'),
                     heatmap_path=os.path.join(args.save_path, 'att_heatmap_image.png'))
# plt.show()