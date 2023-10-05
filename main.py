import time
import numpy as np
from utils import get_logger
import torch
from config import SearchConfig
from models.gp_algorithm import GPAlgorithm
import warnings
import datetime
import os

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    config = SearchConfig()

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d-%H-%M-%S")

    exp_title = f"{timestamp}_{config.data_name}"
    logger = get_logger(os.path.join(config.log_path, f"{exp_title}.log"))

    result_folder = os.path.join(config.result_path, f"{exp_title}")
    os.makedirs(result_folder)

    rounds_experiment = config.rounds_experiment
    logger.info("Experiment_rounds: {}".format(rounds_experiment))
    result = []

    gp_algorithm = GPAlgorithm(config, logger)

    logger.info('===============================================================================================')
    for exp_round in range(1, rounds_experiment):
        logger.info("Current round: {}".format(exp_round))
        round_folder = os.path.join(result_folder, f"{exp_round}")
        os.makedirs(round_folder)

        begin_time = time.process_time()
        test_results = gp_algorithm.run(round_folder)
        end_time = time.process_time()
        train_time = end_time - begin_time
        result.append(test_results)
        logger.info('Test results: {}'.format(test_results))
        logger.info('Train time: {}'.format(train_time))
        logger.info('Complete round: {}'.format(exp_round))
        logger.info('===============================================================================================')
    for ei, r in enumerate(result):
        logger.info("Epoch {} Result: {}".format(ei, r))

    result_np = np.array(result)
    logger.info("Mean: {}".format(np.mean(result_np)))
    logger.info("Std: {}".format(np.std(result_np)))


if __name__ == "__main__":
    main()
