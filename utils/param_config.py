import torch
from utils.utils import Logger
from utils.dataset import Dataset
from utils.partition import KFoldPartition
import yaml
import scipy.io as sio
import numpy as np


class ParamConfig(object):
    def __init__(self, n_run=1, n_kfolds=10, n_agents=25, nrules=10):
        self.model_name = 'fpn'
        self.n_batch = 100
        self.n_epoch = 1000
        self.noise_level = 0.0
        self.n_kfolds = n_kfolds  # Number of folds

        self.n_rules = nrules  # number of rules in stage 1
        self.n_rules_list = []

        self.gni_sigma = 0.0
        self.drop_rate = 0.0

        self.inference = 'nuts'
        self.n_samples = 50
        self.warmup = 12
        self.step_size = 0.001
        self.num_steps = 40
        # svi parameter
        self.lr_svi= 0.01
        self.n_epoch_svi= 200

        self.dataset_list = ['CASP']
        self.dataset_folder = 'hrss'
        self.device = 'cuda:0'

        # set learning rate
        self.lr = 0

        self.log = None

    def config_parse(self, config_name):
        config_dir = f"./configs/{config_name}.yaml"
        config_file = open(config_dir)
        config_content = yaml.load(config_file, Loader=yaml.FullLoader)

        self.model_name = config_content['model']
        self.device = config_content['device']
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.n_batch = config_content['n_batch']
        self.n_epoch = config_content['n_epoch']
        self.n_kfolds = config_content['n_kfolds']
        self.noise_level = config_content['noise_level']
        self.drop_rate = config_content['drop_rate']

        self.n_rules = config_content['n_rules']
        self.lr = config_content['lr']

        self.dataset_list = config_content['dataset_list']
        self.dataset_folder = config_content['dataset_folder']

        # set logger to decide whether write log into files
        if config_content['log_to_file'] == 'false':
            self.log = Logger()
        else:
            self.log = Logger(True, self.dataset_folder)

    def get_dataset(self, dataset_idx=0):
        dataset_name = self.dataset_list[dataset_idx]
        dir_dataset = f"./datasets/{self.dataset_folder}/{dataset_name}.pt"

        load_data = torch.load(dir_dataset)
        dataset_name = load_data['name']
        fea: torch.Tensor = load_data['X']
        gnd: torch.Tensor = load_data['Y']

        if len(gnd.shape) == 1:
            gnd = gnd.unsqueeze(1)

        task = load_data['task']
        dataset = Dataset(fea, gnd, task, dataset_name)

        # set partition strategy
        partition_strategy = KFoldPartition(self.n_kfolds)
        partition_strategy.partition(dataset.gnd, True, 0)
        dataset.set_partition(partition_strategy)
        # dataset.normalize(-1, 1)
        return dataset

    def get_dataset_mat(self, dataset_idx=0):
        dataset_name = self.dataset_list[dataset_idx]
        dir_dataset = f"./datasets/{self.dataset_folder}/{dataset_name}"

        load_data = sio.loadmat(dir_dataset)
        # dataset_name = load_data['name']
        fea: torch.Tensor = torch.tensor(load_data['Data']).float().to(self.device)
        gnd: torch.Tensor = torch.tensor(load_data['Labels'].astype(np.float32)).float().to(self.device)
        fea = torch.permute(fea, [2, 0, 1])

        if len(gnd.shape) == 1:
            gnd = gnd.unsqueeze(1)

        dataset = Dataset(fea, gnd, 'C', dataset_name)

        # set partition strategy
        partition_strategy = KFoldPartition(self.n_kfolds)
        partition_strategy.partition(dataset.gnd, True, 0)
        dataset.set_partition(partition_strategy)
        # dataset.normalize(-1, 1)
        return dataset
