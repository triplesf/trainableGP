import os
from shutil import rmtree
import logging
import datetime
import matplotlib.pyplot as plt
from torchsummary import summary


def mk_dir(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


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


def draw_plt(vis_items):
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y%m%d-%H-%M-%S")
    plt_path = "./plt"

    plt.figure()
    y1 = vis_items["min"]
    y2 = vis_items["max"]
    y3 = vis_items["avg"]
    x = [i for i in range(len(y1))]  # 点的横坐标

    plt.plot(x, y1, marker='o', color='r', label='min')
    plt.plot(x, y2, marker='o', color='b', label='max')
    plt.plot(x, y3, marker='o', color='g', label='average')

    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.legend()

    plt.savefig(os.path.join(plt_path, "{}.png".format(formatted_date)))
    # plt.show()


# def save_model_summary(model, inout_size, file_path):
#     with open(file_path, "w") as file:
#         summary(model, inout_size)

