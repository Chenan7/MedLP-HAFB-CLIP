import torch
import argparse
import time
from datetime import datetime

import dataset_ten as dataset
from model_prompt import CLIP_Model

parser = argparse.ArgumentParser(description='PyTorch clip')
# 训练参数
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--workers', default=0, type=int, help='workers')
# 数据参数
parser.add_argument('--data_path', default='data_ten', type=str, help='start epoch')
parser.add_argument('--model_path', default='ckptsave/prompt/ckpt_best.pth', type=str, help='save path')
# 模型参数
parser.add_argument('--backbone', default='ViT-B/16', type=str)
# coop相关参数
parser.add_argument('--N_CTX', default=12, type=int, help='number of context vectors')
parser.add_argument('--CLASS_TOKEN_POSITION', default='end', type=str, help='middle or end or front')


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    global args
    args = parser.parse_args()

    print(args)

    model = CLIP_Model(args).to(device)
    print(f'model file: {CLIP_Model}')

    print(f'loading model: {args.model_path}')
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'loaded model: {args.model_path}')

    test_loader = torch.utils.data.DataLoader(
        dataset.myDataset(args.data_path, 'test', shuffle=False),
        batch_size=args.bs,
        num_workers=args.workers
    )

    print('Test stage')
    time0 = time.time()
    test_acc = validate(test_loader, model)
    print(' * Test Acc {acc:.4f} '.format(acc=test_acc))
    print(' * Time {time:.4f} '.format(time=time.time() - time0))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def validate(val_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)
            _, _, _, _, logits_per_image = model(image)
            pre = logits_per_image.argmax(dim=-1)
            correct += torch.sum(pre == label).item()
        acc = correct / len(val_loader.dataset)
    return acc


if __name__ == '__main__':
    main()
