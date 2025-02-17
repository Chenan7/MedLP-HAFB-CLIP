import json
import os

import numpy as np
import pandas as pd
import torch
import argparse
import time
from datetime import datetime

from matplotlib import pyplot as plt

import dataset_ten as dataset
from model_final import CLIP_Model
from sklearn.metrics import roc_curve, auc

# 更改不同的模型文件
# from model_base import CLIP_Model     remember delete _, _,
# from model_prompt import CLIP_Model
# from model_final import CLIP_Model

parser = argparse.ArgumentParser(description='PyTorch clip')
# 训练参数
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--workers', default=0, type=int, help='workers')
# 数据参数
parser.add_argument('--data_path', default='data_ten', type=str, help='start epoch')
parser.add_argument('--model_path', default='ckptsave/final/ckpt_best.pth', type=str, help='model path')
parser.add_argument('--save_path', default='test_dir', type=str, help='ROC save path')
# 模型参数
parser.add_argument('--backbone', default='ViT-B/16', type=str)
# coop相关参数
parser.add_argument('--N_CTX', default=12, type=int, help='number of context vectors')
parser.add_argument('--CLASS_TOKEN_POSITION', default='end', type=str, help='middle or end or front')


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    global args, label_names
    args = parser.parse_args()

    print(args)
    with open(os.path.join(args.data_path, 'label_names.json')) as f:
        label_names = json.load(f)

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
            #_, _, logits_per_image = model(image)
            _, _, _, _, logits_per_image = model(image)  # baseline 只有三个输入 prompt & final有五个输入
            logits_per_image = logits_per_image.softmax(dim=-1)
            y_pred.extend(logits_per_image.tolist())
            y_true.extend(label.tolist())

        y_pred = np.array(y_pred)
        df = pd.DataFrame()
        # 每个分类的预测结果概率
        df['y_test'] = y_true
        for i in range(10):
            df[f'pre_score{i}'] = y_pred[:, i]

        y_list = df['y_test'].to_list()
        pre_list = [np.array(df[f'pre_score{i}']) for i in range(10)]

        label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        colors1 = ["r", "b", "g", 'gold', 'pink', 'y', 'c', 'm', 'orange', 'chocolate']
        linestyles = ["-", "--", ":", "-", "--", ":", "-", "--", ":", "-"]

        plt.figure(figsize=(12, 6), facecolor='w')
        for i in range(10):
            fpr, tpr, _ = roc_curve(y_list, pre_list[i], pos_label=i)
            roc_auc = auc(fpr, tpr)

            # 将 linewidth 从 3 改为 1.5，使线条变细
            plt.plot(fpr, tpr, color=colors1[i], linestyle=linestyles[i], linewidth=1.5,
                     label=f"class {label_names[i]}: area = {roc_auc:.2f}")

        # 添加对角线，线条也变细
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5, label="Random guess (area = 0.50)")

        plt.xlabel('False Positive Rate',fontsize=16)
        plt.ylabel('True Positive Rate',fontsize=16)
        plt.grid()
        plt.legend(loc='lower right', fontsize=16)  # 图例位置和字体大小
        plt.title("ROC Curve for Each Class",fontsize=18)
        save_path = os.path.join(args.save_path, f'ROC_curve_final.png')     # 每个模型的保存地址，老是忘了改
        plt.savefig(save_path, dpi=300)



if __name__ == '__main__':
    main()



