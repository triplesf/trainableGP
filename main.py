import time
import numpy as np
from common import init_log
import torch
from dataset.dataset import prepare_dataset, read_dataset_info
from config.main_config import SearchConfig
from models.tree_structure import initialize_standard_operations, initialize_darts_operations, gp_process
import warnings

warnings.filterwarnings('ignore')


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    config = SearchConfig()
    data_name = config.data_name
    data_path = config.data_path

    # logger = init_log(data_name)
    train_dataset, val_dataset, test_dataset, all_train_dataset = prepare_dataset(data_path, data_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    all_train_loader = torch.utils.data.DataLoader(all_train_dataset, batch_size=16, shuffle=False)
    rounds_experiment = config.rounds_experiment

    logger = init_log(data_name)
    num_classes = read_dataset_info(data_path, data_name, logger)
    result = []
    if config.network_operations == "standard":
        toolbox = initialize_standard_operations(config, device, train_loader, val_loader, test_loader,
                                                 all_train_loader, num_classes=num_classes)
    elif config.network_operations == "darts":
        toolbox = initialize_darts_operations(config, device, train_loader, val_loader, test_loader,
                                                 all_train_loader, num_classes=num_classes)
    else:
        raise Exception("Invalid Network Operations!")

    # logger.info("Data_name: {}".format(data_name))
    logger.info("GP Info:")
    logger.info("Population: {}".format(config.population))
    logger.info("Generations: {}".format(config.generations))
    logger.info("Experiment_rounds: {}".format(rounds_experiment))
    logger.info("cxProb: {}, mutProb: {}, elitismProb: {}, maxDepth: {}".format(config.cxProb, config.mutProb,
                                                                                config.elitismProb, config.maxDepth))
    logger.info('===============================================================================================')
    for exp_round in range(1, rounds_experiment):
        logger.info("Current round: {}".format(exp_round))
        beginTime = time.process_time()

        testResults = gp_process(config=config, toolbox=toolbox, logger=logger)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        # logger.info('Best individual {}'.format(hof[0]))
        result.append(testResults)
        logger.info('Test results: {}'.format(testResults))
        logger.info('Train time: {}'.format(trainTime))
        logger.info('Complete round: {}'.format(exp_round))
        logger.info('===============================================================================================')
    for ei, r in enumerate(result):
        logger.info("Epoch {} Result: {}".format(ei, r))

    result_np = np.array(result)
    # logger.info("Average: {}".format(round(sum(result) / len(result)), 2))
    logger.info("Mean: {}".format(np.mean(result_np)))
    logger.info("Std: {}".format(np.std(result_np)))


if __name__ == "__main__":
    main()
