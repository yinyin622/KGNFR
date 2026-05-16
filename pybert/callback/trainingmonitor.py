# encoding:utf-8
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 
from ..common.tools import load_json
from ..common.tools import save_json
plt.switch_backend('agg')


class TrainingMonitor():
    def __init__(self, file_dir, arch, add_test=False, time=''):
        '''
        :param startAt: 重新开始训练的epoch点
        '''
        if not isinstance(file_dir, Path):
            file_dir = Path(file_dir)

        # 构建新的目录：file_dir/time
        file_dir = file_dir / time  # 关键修改：拼接时间戳
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.H = {}
        self.add_test = add_test
        self.json_path = file_dir / (arch + "_training_monitor.json")

        # 初始化 TensorBoard 的 SummaryWriter
        # 存储路径 
        self.writer = SummaryWriter(log_dir=file_dir / "../../runs")

        # 用于绘图的路径
        self.paths = {}

    def reset(self,start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():

                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs={}):
        for (k, v) in logs.items():
            # 尝试从self.H字典中获取键k对应的值，并将其存储在变量l中
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v, (float, np.floating)):
                v = round(float(v), 4)
            l.append(v)
            self.H[k] = l

        # 写入 JSON 文件
        if self.json_path is not None:
            save_json(data = self.H,file_path=self.json_path)

        # 写入 TensorBoard
        epoch = len(self.H['loss']) - 1  # 当前 epoch 索引（从 0 开始）
        for key, value in logs.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(key, value, global_step=epoch)


        # 保存train图像
        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key.upper()}') for key in self.H.keys()}

        if len(self.H["loss"]) > 1:
            keys = ['auc','precision', 'recall', 'hamming_score', 'hamming_loss', 'f1_score']
            for key in keys:
                if key not in self.H:
                    continue  # 跳过未记录的训练指标

                train_key = key
                valid_key = f"valid_{key}"
                test_key = f"test_{key}"

                # 检查长度是否一致（防止不同步）
                min_len = len(self.H[train_key])
                if valid_key in self.H:
                    min_len = min(min_len, len(self.H[valid_key]))
                else:
                    continue  # 如果没有 valid 指标，跳过绘图（或只画 train）

                if self.add_test and test_key in self.H:
                    min_len = min(min_len, len(self.H[test_key]))

                N = np.arange(0, min_len)
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, self.H[train_key][:min_len], label=f"train_{key}")
                if valid_key in self.H:
                    plt.plot(N, self.H[valid_key][:min_len], label=f"valid_{key}")
                if self.add_test and test_key in self.H:
                    plt.plot(N, self.H[test_key][:min_len], label=f"test_{key}")

                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[train_key])}]")
                plt.savefig(str(self.file_dir / (self.arch + f'_{key.upper()}.png')))
                plt.close()


    def close(self):
        """关闭 writer，确保日志写入磁盘"""
        if hasattr(self, 'writer'):
            self.writer.close()