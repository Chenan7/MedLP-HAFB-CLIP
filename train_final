import os
import random
import numpy as np

from model_final import CLIP_Model

import torch
import torch.nn as nn

import argparse
import dataset_ten as dataset
import time
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch clip')
# 训练参数
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--opt_eps', default=1e-8, type=float, help='optimizer eps')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--workers', default=0, type=int, help='workers')
parser.add_argument('--print_freq', default=10, type=int, help='print freq.')
# 损失参数
parser.add_argument('--clip_loss', default='true', type=str, help='use clip loss')
parser.add_argument('--clip_weight', default=1.0, type=float, help='clip loss weight')
parser.add_argument('--cls_loss', default='false', type=str, help='use cls loss')
parser.add_argument('--cls_weight', default=10.0, type=float, help='cls loss weight')
parser.add_argument('--cls59_weight', default=0.1, type=float, help='weight')
# 数据参数
parser.add_argument('--data_path', default='data_ten', type=str, help='start epoch')
parser.add_argument('--save_path', default='checkpoint', type=str, help='save path')
# 模型参数
parser.add_argument('--backbone', default='ViT-B/16', type=str)
# coop相关参数
parser.add_argument('--N_CTX', default=8, type=int, help='number of context vectors')
parser.add_argument('--CLASS_TOKEN_POSITION', default='end', type=str, help='middle or end or front')


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    global args, best_acc
    best_acc = 0.0
    
    args = parser.parse_args()
    args.seed = round(time.time())
    args.clip_loss = args.clip_loss == 'true'
    args.cls_loss = args.cls_loss == 'true'

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    model = CLIP_Model(args).to(device)
    print(f'model file: {CLIP_Model}')
    print(f'total_params: {sum(p.numel() for p in model.parameters())}')
    print(f'trainable_params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    ce_loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay,
                                 eps=args.opt_eps)

    train_loader = torch.utils.data.DataLoader(
        dataset.myDataset(args.data_path, 'train', shuffle=True),
        batch_size=args.bs,
        num_workers=args.workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset.myDataset(args.data_path, 'valid', shuffle=False),
        batch_size=args.bs,
        num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset.myDataset(args.data_path, 'test', shuffle=False),
        batch_size=args.bs,
        num_workers=args.workers
    )

    for epoch in range(args.epochs):
        train(train_loader, model, ce_loss, optimizer, epoch)
        print('Val stage')
        start_time = time.time()
        val_acc = validate(val_loader, model)
        print(' * Acc {acc:.4f} '.format(acc=val_acc))
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        print(' * best Acc {acc:.4f} '.format(acc=best_acc))
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(args.save_path, 'ckpt.pth'))
        end_time = time.time()
        print(' * Time {time:.4f} '.format(time=end_time - start_time))
        
        if is_best:
            torch.save(state, os.path.join(args.save_path, 'ckpt_best.pth'))
            print('Test stage')
            test_acc = validate(test_loader, model)
            print(' * Test Acc {acc:.4f} '.format(acc=test_acc))
            print(' * Time {time:.4f} '.format(time=time.time() - end_time))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def train(train_loader, model, ce_loss, optimizer, epoch):
    losses = AverageMeter()
    losses_clip = AverageMeter()
    losses_cls59 = AverageMeter()
    losses_cls = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()
    
    for i, (image, label) in enumerate(train_loader):
        bs = image.shape[0]
        data_time.update(time.time() - end)
        image = image.to(device)
        label = label.to(device)

        logits_per_image_clip, logits_per_text_clip,\
        logits_per_image_59_cls, class_label_list, logits_per_image_cls = model(image, label, args.clip_loss, args.cls_loss)
        loss_labels = torch.arange(bs).to(device)
        loss_image = torch.tensor(0.0, dtype=torch.float32, device=device) if logits_per_image_clip is None else ce_loss(logits_per_image_clip, loss_labels)
        loss_text = torch.tensor(0.0, dtype=torch.float32, device=device) if logits_per_text_clip is None else ce_loss(logits_per_text_clip, loss_labels)
        loss_clip = (loss_image + loss_text) / 2
        loss_clip = loss_clip * args.clip_weight
        losses_clip.update(loss_clip.item(), bs)

        # len_59_cls
        if logits_per_image_59_cls is None:
            loss_59_cls = torch.tensor(0.0, dtype=torch.float32,device=device)
        else:
            len_59_cls = len(logits_per_image_59_cls)
            loss_59_cls = ce_loss(logits_per_image_59_cls, torch.tensor(class_label_list, device=device))
            loss_59_cls = loss_59_cls * args.cls59_weight
            losses_cls59.update(loss_59_cls.item(), len_59_cls)

        loss_cls = torch.tensor(0.0, dtype=torch.float32, device=device) if logits_per_image_cls is None else ce_loss(logits_per_image_cls, label)  # [bs, 2] 和 [bs]
        loss_cls = loss_cls * args.cls_weight
        losses_cls.update(loss_cls.item(), bs)

        loss = loss_clip + loss_cls + loss_59_cls
        losses.update(loss.item(), bs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_clip {loss_clip.val:.4f} ({loss_clip.avg:.4f})\t'
                  'Loss_cls59 {loss_cls59.val:.4f} ({loss_cls59.avg:.4f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_clip=losses_clip,
                   loss_cls59=losses_cls59, loss_cls=losses_cls))


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


if __name__ == '__main__':
    main()
