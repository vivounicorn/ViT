'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inference import Cifar10Inference
from models.teachers.resnet import ResNet50
from utils.config_utils import Config
from utils.data_utils import data_loader


def setup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = ResNet50()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if os.path.exists('./checkpoint/resnet50_ckpt.pth'):
        checkpoint = torch.load('./checkpoint/resnet50_ckpt.pth')
        net.load_state_dict(checkpoint, False)

    return net, device


# Training
def train(trainer_name, log_dir, epoch):
    net, device = setup()
    writer = SummaryWriter(os.path.join(log_dir, trainer_name))

    # 加载训练数据和测试数据
    train_loader, test_loader = data_loader(244, 32, 32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print('\nEpoch: %d' % epoch)
    net.zero_grad()

    # 计算平均损失
    loss_sum = 0
    loss_count = 0
    # 当前迭代步数
    current_step = 0
    # 当前最佳精度
    current_best_acc = 0

    while True:
        net.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training Progress [x / x Total Steps] {loss=x.x}",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            # 获取一个batch，并把数据发送到相应设备上（如：GPU卡）
            batch = tuple(t.to(device) for t in batch)
            features, labels = batch
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            # 全局平均损失
            loss_sum += loss.item()
            loss_count += 1
            # 梯度正则化，缓解过拟合
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            # 执行一步最优化方法
            optimizer.step()
            # 执行学习率调整策略
            scheduler.step()
            # 梯度清零
            optimizer.zero_grad()
            # 存储当前迭代步数
            current_step += 1

            epoch_iterator.set_description(
                "Training Progress [%d / %d Total Steps] {loss=%2.5f}" % (
                    current_step, epoch, loss_sum / loss_count)
            )

            writer.add_scalar("train/loss", scalar_value=loss_sum / loss_count, global_step=current_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=current_step)

            # 每迭代若干步后做测试集验证
            if current_step % 100 == 0:
                accuracy = test(net, device, writer, criterion, test_loader, current_step)
                # 测试集上表现好的模型被存储，并更新当前最佳精度
                if current_best_acc < accuracy:
                    model_bin = net.module if hasattr(net, 'module') else net
                    torch.save(model_bin.state_dict(), './checkpoint/resnet50_ckpt.pth')
                    current_best_acc = accuracy
                # 接着训练
                net.train()

            if current_step % epoch == 0:
                break
        loss_sum = 0
        loss_count = 0
        if current_step % epoch == 0:
            break

    writer.close()
    print("Best Accuracy: \t%f" % current_best_acc)
    print("***** End Training *****")


def test(net, device, writer, criterion, test_loader, current_step):
    print("\r\n***** Running Testing *****")
    t_total = len(test_loader)
    loss_sum = 0.0
    loss_count = 0

    net.eval()
    preds_list, label_list = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Testing progress [x / x Total Steps] {loss=x.x}",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    count = 0
    for step, batch in enumerate(epoch_iterator):
        count += 1
        batch = tuple(t.to(device) for t in batch)
        features, labels = batch
        with torch.no_grad():
            outputs = net(features)
            eval_loss = criterion(outputs, labels)
            # 全局平均损失,tensor是标量，用item方法取出
            loss_sum += eval_loss.item()
            loss_count += 1
            # 返回预测分类标号
            _, preds = outputs.max(1)

        # 把预测的结果和标注从gpu tensor转换为cpu tensor后再转换为numpy数组.(注：不能直接从gpu tensor转为numpy数组)
        preds_list.append(preds.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())
        epoch_iterator.set_description(
            "Testing Progress [%d / %d Total Steps] {loss=%2.5f}" % (count, t_total, loss_sum / loss_count))

    # 横向拼接list里的所有numpy数组
    preds_list = np.hstack(preds_list)
    label_list = np.hstack(label_list)
    accuracy = (preds_list == label_list).mean()

    print("\n")
    print("Testing Results")
    print("Current Steps: %d" % current_step)
    print("Average Testing Loss: %2.5f" % (loss_sum / loss_count))
    print("Testing Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=current_step)
    return accuracy


def unit_test():
    train('resnet50', './logs/resnet50', 10000)

    models_cfg = Config()
    models_cfg.load_config(os.path.join(os.path.dirname(__file__), 'config/config.ini'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet50()
    net.to(device)
    model_path = '/home/dell/PycharmProjects/ViT/checkpoint/resnet50_ckpt.pth'
    ci = Cifar10Inference(net, model_path, models_cfg)
    img_path = '/home/dell/PycharmProjects/ViT/data/demo/dog.jpeg'
    ci.classify(img_path)
