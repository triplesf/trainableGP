""" Config class for search/augment """
import argparse
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--cross_validation', action='store_true', default=False, help='Perform cross-validation')
        parser.add_argument('--classes_number', type=int, default=2, help='Number of classes')
        parser.add_argument('--cxProb', type=float, default=0.5, help='Crossover probability')
        parser.add_argument('--data_name', type=str, default="f1_ours", help='Data name')
        parser.add_argument('--data_path', type=str, default="dataset/processed/f1", help='Data path')
        parser.add_argument('--dataset_source', choices=['numpy', 'torch'], type=str,
                            default="torch",
                            help='Choose between "numpy" or "torch" for dataset source')
        parser.add_argument('--elitismProb', type=float, default=0.05, help='Elitism probability')
        parser.add_argument('--epochs', type=int, default=20, help='# of training epochs')
        parser.add_argument('--gpus', default='0', help='GPU device ids separated by comma. '
                            '`all` indicates use all GPUs')
        parser.add_argument('--generations', type=int, default=50, help='Number of generations')
        parser.add_argument('--initialMaxDepth', type=int, default=6, help='Initial maximum depth')
        parser.add_argument('--initialMinDepth', type=int, default=2, help='Initial minimum depth')
        parser.add_argument('--image_mode', type=str, default="rgb", help='')
        parser.add_argument('--log_path', type=str, default="log", help='Log file path')
        parser.add_argument('--maxDepth', type=int, default=8, help='Maximum depth')
        parser.add_argument('--mutProb', type=float, default=0.6, help='Mutation probability')
        parser.add_argument('--network_operations', choices=['standard', 'darts', 'single'], type=str,
                            default="standard",
                            help='Choose between "standard", "darts" or "single" for network operations')
        parser.add_argument('--num_hidden_layers', type=int, default=64, help='Number of hidden layers')
        parser.add_argument('--population', type=int, default=100, help='Population')
        parser.add_argument('--result_path', type=str, default="results", help='Result folder path')
        parser.add_argument('--rounds_experiment', type=int, default=10, help='Number of rounds of experiments')
        parser.add_argument('--samples_per_class', type=int, default=20, help='Number of samples per class')
        parser.add_argument('--seed', type=int, default=0, help='Random seed')
        parser.add_argument('--test_train_epochs', type=int, default=400, help='# of training epochs')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # self.data_path = './data/'
        # self.path = os.path.join('searchs', self.name)
        # self.plot_path = os.path.join(self.path, 'plots')
        # self.gpus = parse_gpus(self.gpus)

