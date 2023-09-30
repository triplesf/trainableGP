import time
import numpy as np
from common import init_log
import torch
from dataset import prepare_dataset
from config.main_config import SearchConfig
from models.tree_structure import init_structure, gp_process
import warnings

warnings.filterwarnings('ignore')


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    config = SearchConfig()
    data_name = config.data_name
    # logger = init_log(data_name)
    train_dataset, val_dataset, test_dataset, all_train_dataset = prepare_dataset("dataset", data_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    all_train_loader = torch.utils.data.DataLoader(all_train_dataset, batch_size=16, shuffle=False)

    epoch_num = 10

    logger = init_log(data_name)
    result = []
    toolbox = init_structure(config, device, train_loader, val_loader, test_loader, all_train_loader)
    logger.info("data_name: {}".format(data_name))
    logger.info("population: {}".format(config.population))
    logger.info("generations: {}".format(config.generations))
    logger.info("epochNum: {}".format(epoch_num))
    logger.info("cxProb: {}, mutProb: {}, elitismProb: {}, maxDepth: {}".format(config.cxProb, config.mutProb,
                                                                                config.elitismProb, config.maxDepth))
    logger.info('===============================================================================================')
    for epoch in range(epoch_num):
        logger.info("Start {} epoch".format(epoch))
        beginTime = time.process_time()

        testResults = gp_process(config=config, toolbox=toolbox, logger=logger)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        # logger.info('Best individual {}'.format(hof[0]))
        result.append(testResults)
        logger.info('Test results  {}'.format(testResults))
        logger.info('Train time  {}'.format(trainTime))
        logger.info('End {} epoch'.format(epoch))
        logger.info('===============================================================================================')
    for ei, r in enumerate(result):
        logger.info("Epoch {} Result: {}".format(ei, r))

    result_np = np.array(result)
    # logger.info("Average: {}".format(round(sum(result) / len(result)), 2))
    logger.info("Mean: {}".format(np.mean(result_np)))
    logger.info("Std: {}".format(np.std(result_np)))


if __name__ == "__main__":
    main()
