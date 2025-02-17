import torch
import argparse
from datetime import datetime
from thop import profile

from model_base import CLIP_Model
# 更改不同的模型文件
# from model_base import CLIP_Model     remember delete _, _,
# from model_prompt import CLIP_Model
# from model_final import CLIP_Model



parser = argparse.ArgumentParser(description='PyTorch clip')
# 数据参数
parser.add_argument('--data_path', default='data_ten', type=str, help='start epoch')
parser.add_argument('--model_path', default='ckptsave/baseline/ckpt_best.pth', type=str, help='save path')
# 模型参数
parser.add_argument('--backbone', default='ViT-B/16', type=str)
# coop相关参数
parser.add_argument('--N_CTX', default=12, type=int, help='number of context vectors')
parser.add_argument('--CLASS_TOKEN_POSITION', default='end', type=str, help='middle or end or front')

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    args = parser.parse_args()

    print(args)

    model = CLIP_Model(args).to(device)
    print(f'model file: {CLIP_Model}')
    print(f'total_params: {sum(p.numel() for p in model.parameters())}')
    print(f'trainable_params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    input = torch.zeros((1, 3, 224, 224)).to(device)
    flops, _ = profile(model.to(device), inputs=(input,))
    print(f'FLOPS: {flops}')

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()



#----------------------------baseline结果---------------------------------------------------------
# total_params: 149620737
# trainable_params: 86192640
# FLOPS: 30648041472.0
#----------------------------pormpt结果-----------------------------------------------------------
# total_params: 149626881
# trainable_params: 86198784
# FLOPS: 88781094912.0
#----------------------------final结果------------------------------------------------------------
# total_params: 389102343
# trainable_params: 325674246
# FLOPS: 89020563456.0