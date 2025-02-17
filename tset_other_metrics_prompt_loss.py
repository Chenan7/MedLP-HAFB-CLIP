import torch
import argparse
import time
from datetime import datetime

from sklearn.metrics import classification_report

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
    validate(test_loader, model)
    print(' * Time {time:.4f} '.format(time=time.time() - time0))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def validate(val_loader, model):
    model.eval()
    with torch.no_grad():
        y_pred, y_true = [], []
        for i, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)
            _, _, _, _, logits_per_image = model(image)
            pre = logits_per_image.argmax(dim=-1)
            y_pred.extend(pre.tolist())  # 取索引1，代表是假的概率
            y_true.extend(label.tolist())
        measure_result = classification_report(y_true, y_pred)
        print('measure_result = \n', measure_result)


if __name__ == '__main__':
    main()

#----------------------------baseline结果---------------------------------------------------------
# 2025-02-02 15:23:05
# cuda
# Namespace(CLASS_TOKEN_POSITION='end', N_CTX=12, backbone='ViT-B/16', bs=16, data_path='data_ten', model_path='ckptsave/prompt/ckpt_best.pth', workers=0)
# Initializing a generic context
# Initial context: "X X X X X X X X X X X X"
# Number of context words (tokens): 12
# model file: <class 'model_prompt.CLIP_Model'>
# loading model: ckptsave/prompt/ckpt_best.pth
# loaded model: ckptsave/prompt/ckpt_best.pth
# Test stage
# measure_result =
#                precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00        92
#            1       1.00      0.98      0.99        89
#            2       1.00      1.00      1.00        86
#            3       1.00      1.00      1.00        74
#            4       0.99      0.98      0.99       115
#            5       0.93      0.93      0.93        80
#            6       0.99      1.00      0.99        97
#            7       1.00      1.00      1.00       103
#            8       1.00      1.00      1.00       120
#            9       0.88      0.91      0.89        65
#
#     accuracy                           0.98       921
#    macro avg       0.98      0.98      0.98       921
# weighted avg       0.98      0.98      0.98       921
#
#  * Time 22.7259
# 2025-02-02 15:23:32
#-------------------------------------------------------------------------------------


