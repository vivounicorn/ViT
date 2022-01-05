import math
import os

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from utils.data_utils import data_loader, find_newest_model


class Trainer(object):
    """
    training model.
    """

    def __init__(self, trainer_name, model, config):
        """
        trainer initialize.
        :param trainer_name: trainer's name.
        :param model: vit model.
        :param config: configuration file.
        """
        # 训练器名字
        self.trainer_name = trainer_name
        self.model = model
        self.config = config

        # fine-tuning模型所在目录
        self.fine_tuning_dir = os.path.join(os.path.dirname(__file__), 'finetuning')
        # 本次训练checkpoint目录
        self.check_point_dir = os.path.join(os.path.dirname(__file__), 'checkpoint')
        # 日志文件所在根目录
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')

        os.makedirs(self.fine_tuning_dir, exist_ok=True)
        os.makedirs(os.path.join(self.check_point_dir, self.trainer_name), exist_ok=True)
        # 本次训练日志文件所在位置
        os.makedirs(os.path.join(self.log_dir, self.trainer_name), exist_ok=True)

        # 用于在tensorboard上查看训练进度。Usage：tensorboard --logdir=xxx --port 8123
        self.writer = SummaryWriter(os.path.join(self.log_dir, self.trainer_name))
        # 打印训练器配置参数
        self.trainer_paras_summary()

    def trainer_paras_summary(self):
        """
        trainer‘s parameter summary.’
        :return: None.
        """
        print("\033[34m***** Trainer Parameters *****\033[0m")
        print("  batch size of training:\033[31m%d\033[0m" % self.config.items.train_batch_size)
        print("  batch size of testing:\033[31m%d\033[0m" % self.config.items.test_batch_size)
        print("  learning rate:\033[31m%f\033[0m" % self.config.items.learning_rate)
        print("  number of training steps:\033[31m%d\033[0m" % self.config.items.num_steps)
        print("  path of pretrained model:\033[31m%s\033[0m" % self.config.items.pretrained_model)
        print("  warmup steps:\033[31m%d\033[0m" % self.config.items.warmup_steps)
        print("  testing every:\033[31m%d\033[0m steps" % self.config.items.test_epoch)
        print("\033[34m***** End *****\033[0m")

    def load_model(self, is_load=True):
        """
        loading pretrained model.
        :param is_load: whether load model.
        :return: True: success False: otherwise.
        """
        if is_load:
            load_path = os.path.join(self.fine_tuning_dir, self.trainer_name)
            if not os.path.exists(load_path):
                return False
            name, file = find_newest_model(load_path)
            if name is not None:
                try:
                    self.model.load_state_dict(torch.load(file))
                    print("\033[34m successfully loaded model:%s from path:%s from path \033[0m" % (name, file))
                    return True
                except Exception as e:
                    print(e)

        return False

    def save_model(self):
        """
        saving model.
        :return: None.
        """
        model_bin = self.model.module if hasattr(self.model, 'module') else self.model
        model_chkpt = os.path.join(self.check_point_dir, self.trainer_name, "%s_chkpt.bin" % self.trainer_name)
        torch.save(model_bin.state_dict(), model_chkpt)
        print("Trainer [%s] Saved model checkpoint to [%s]" % (self.trainer_name, self.check_point_dir))

    def test(self, test_loader, current_step):
        """
        testing model every few steps
        :param test_loader: testing data loader.
        :param current_step: current training steps.
        :return: testing accuracy.
        """

        print("\r\n***** Running Testing *****")
        t_total = len(test_loader)
        loss_sum = 0.0
        loss_count = 0

        self.model.eval()
        preds_list, label_list = [], []
        epoch_iterator = tqdm(test_loader,
                              desc="Testing progress [x / x Total Steps] {loss=x.x}",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        count = 0
        for step, batch in enumerate(epoch_iterator):
            count += 1
            batch = tuple(t.to(self.config.items.device) for t in batch)
            features, labels = batch
            with torch.no_grad():
                eval_loss, obj_res = self.model(features, labels)
                # 全局平均损失,tensor是标量，用item方法取出
                loss_sum += eval_loss.item()
                loss_count += 1
                # 返回预测分类标号
                preds = torch.argmax(obj_res, dim=-1)

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

        self.writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=current_step)
        return accuracy

    def train(self):
        """
        training the model
        :return: None
        """

        self.load_model()
        # 加载训练数据和测试数据
        train_loader, test_loader = data_loader(self.config.items.img_size,
                                                self.config.items.train_batch_size,
                                                self.config.items.test_batch_size)
        # 选择一个一阶最优化方法
        if self.config.items.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.config.items.learning_rate)
        elif self.config.items.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(),
                                            lr=self.config.items.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config.items.learning_rate,
                                        momentum=0.9)

        # 总的迭代步数
        t_total = self.config.items.num_steps
        # 以此步数做学习率分段函数
        warmup_steps = self.config.items.warmup_steps
        # 随着迭代步数增加，调整学习率的策略(cosine法|S)
        #        *
        #       *   *
        #      *      *
        #     *        *
        #    *          *
        #   *            *
        #  *               *
        # *                   *
        # *                       *   *    *
        scheduler = LambdaLR(optimizer=optimizer,
                             lr_lambda=lambda step: float(step) / warmup_steps if step < warmup_steps else 0.5 * (
                                     math.cos(math.pi * 0.6 * 2.0 * (step - warmup_steps) / (
                                             t_total - warmup_steps)) + 1.0))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        print("\r\n***** Running training *****")

        # 模型所有参数梯度值初始化为0
        self.model.zero_grad()
        # 计算平均损失
        loss_sum = 0
        loss_count = 0
        # 当前迭代步数
        current_step = 0
        # 当前最佳精度
        current_best_acc = 0

        while True:
            self.model.train()
            # 初始化进度条
            epoch_iterator = tqdm(train_loader,
                                  desc="Training Progress [x / x Total Steps] {loss=x.x}",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)

            for step, batch in enumerate(epoch_iterator):
                # 获取一个batch，并把数据发送到相应设备上（如：GPU卡）
                batch = tuple(t.to(self.config.items.device) for t in batch)
                # 特征与标注数据
                features, labels = batch
                loss, _ = self.model(features, labels)
                # 自动反向传播求梯度
                loss.backward()
                # 全局平均损失
                loss_sum += loss.item()
                loss_count += 1
                # 梯度正则化，缓解过拟合
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
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
                        current_step, t_total, loss_sum / loss_count)
                )

                self.writer.add_scalar("train/loss", scalar_value=loss_sum / loss_count, global_step=current_step)
                self.writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=current_step)

                # 每迭代若干步后做测试集验证
                if current_step % self.config.items.test_epoch == 0:
                    accuracy = self.test(test_loader, current_step)
                    # 测试集上表现好的模型被存储，并更新当前最佳精度
                    if current_best_acc < accuracy:
                        self.save_model()
                        current_best_acc = accuracy
                    # 接着训练
                    self.model.train()

                if current_step % t_total == 0:
                    break
            loss_sum = 0
            loss_count = 0
            if current_step % t_total == 0:
                break

        self.writer.close()
        print("Best Accuracy: \t%f" % current_best_acc)
        print("***** End Training *****")
