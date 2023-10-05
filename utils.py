import os
from shutil import rmtree
import logging
import datetime
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
from io import StringIO
from torchsummary import summary
import subprocess
import time


def mk_dir(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def print_dynamic_gpu_utilization(interval_seconds=2):
    try:
        while True:
            # 使用nvidia-smi命令获取GPU利用率
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 检查命令是否成功运行
            if result.returncode == 0:
                gpu_utilization = result.stdout.strip()
                # 清屏（可选）
                subprocess.run(['clear'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
                print(f"GPU利用率: {gpu_utilization}%")
            else:
                print("无法获取GPU利用率")

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        # 捕获Ctrl+C，退出循环
        pass
    except Exception as e:
        print(f"获取GPU利用率时出错: {str(e)}")


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('trainableGP')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def init_log(dataSetName):
    now = datetime.datetime.now()
    # 格式化日期和时间
    formatted_date = now.strftime("%Y%m%d-%H-%M-%S")
    log_path = "./log"
    logger = get_logger(os.path.join(log_path, "{}-{}.log".format(formatted_date, dataSetName)))
    return logger


@contextmanager
def string_io_redirected():
    # 保存原始的 sys.stdout
    original_stdout = sys.stdout

    # 创建一个 StringIO 对象，用于捕获输出
    string_io = StringIO()

    # 将 sys.stdout 重定向到 StringIO 对象
    sys.stdout = string_io

    try:
        yield string_io
    finally:
        # 恢复原始的 sys.stdout
        sys.stdout = original_stdout


# 捕获一个 print 输出
def write_summary(summary_path, model, input_size):
    original_stdout = sys.stdout
    with string_io_redirected() as captured_output:
        summary(model, input_size)
    # 恢复原始的 sys.stdout
    sys.stdout = original_stdout
    captured_output = captured_output.getvalue()
    with open(os.path.join(summary_path, "summary.txt"), "w") as f:
        f.write('\n' + captured_output)

# def save_model_summary(model, inout_size, file_path):
#     with open(file_path, "w") as file:
#         summary(model, inout_size)


if __name__ == '__main__':
    print_dynamic_gpu_utilization()

