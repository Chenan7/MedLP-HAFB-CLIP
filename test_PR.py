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
from model_prompt import CLIP_Model
from sklearn.metrics import precision_recall_curve, average_precision_score

# 更改不同的模型文件
# from model_base import CLIP_Model     remember delete _, _,
# from model_prompt import CLIP_Model
# from model_final import CLIP_Model


df = pd.DataFrame()
# pre_score = model.predict_proba(X_test)
# df['y_test'] = Y_test.to_list()

parser = argparse.ArgumentParser(description='PyTorch clip')
# 训练参数
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--workers', default=0, type=int, help='workers')
# 数据参数
parser.add_argument('--data_path', default='data_ten', type=str, help='start epoch')
parser.add_argument('--model_path', default='ckptsave/prompt/ckpt_best.pth', type=str, help='model path')
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

# if i > 2:
#     break

def validate(val_loader, model, num_classes=10):
    model.eval()
    with torch.no_grad():
        y_pred, y_true = [], []

        for i, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)
            #_, _, outputs = model(image)
            _, _, _, _, outputs = model(image)   #baseline 只有三个输入 prompt & final有五个输入
            outputs = outputs.softmax(dim=-1)  # 计算每个类别的概率
            y_pred.extend(outputs.tolist())
            y_true.extend(label.tolist())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # 初始化绘图所需变量
        label_names = [str(i) for i in range(num_classes)]
        colors1 = ["r", "b", "g", 'gold', 'pink', 'y', 'c', 'm', 'orange', 'chocolate']
        linestyles = ["-", "--", ":", "-", "--", ":", "-", "--", ":", "-"]

        plt.figure(figsize=(12, 6), facecolor='w')

        # 动态调整文本位置
        # y_text_positions = np.linspace(0.80, 0.05, num_classes)
        for i in range(num_classes):
            # 计算每个类别的 Precision-Recall 曲线
            precision, recall, _ = precision_recall_curve(
                (y_true == i).astype(int), y_pred[:, i]
            )
            avg_precision = average_precision_score(
                (y_true == i).astype(int), y_pred[:, i]
            )


            # 绘制曲线
            plt.plot(
                recall,
                precision,
                color=colors1[i % len(colors1)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,
                label=f"class {label_names[i]}: area = {avg_precision:.2f}",
            )

        # 设置图像标题和保存路径
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.grid()
        plt.legend(loc="lower left", fontsize=16)
        plt.title("Precision-Recall Curve for Each Class", fontsize=18)
        save_path = os.path.join(args.save_path, "PR_curve_prompt_loss.png")        # 每个模型的保存地址，老是忘了改
        plt.savefig(save_path, dpi=300)
        # plt.show()



if __name__ == '__main__':
    main()