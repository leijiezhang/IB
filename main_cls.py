from utils.param_config import ParamConfig
from utils.loss_utils import RMSELoss, LikelyLoss
from models.model_run import run_model, run_model_cls
import torch
import os
import scipy.io as io
import argparse
from models.fpn_models import *
from models.eegnet import EEGNet
from sklearn.decomposition import PCA
from models.vit import ViT
from models.resnet import resnet18


# Dataset configuration
# Dataset
parser = argparse.ArgumentParser()
parser.add_argument(
        "--d",
        type=str,
        default="getlost",
        help="the name of dataset",
    )
parser.add_argument(
        "--model",
        type=str,
        default="eegnet",
        help="the name of model",
    )
# parser.add_argument(
#         "--nl",
#         type=float,
#         default=None,
#         help="the name of dataset",
#     )
args = parser.parse_args()
# init the parameters statlib_calhousing_config
param_config = ParamConfig()
param_config.config_parse("06ib_config")

# if args.nl is not None:
#     param_config.noise_level = args.nl

param_config.log.info(f"dataset : {param_config.dataset_folder}")
param_config.log.info(f"rule number : {param_config.n_rules}")
param_config.log.info(f"batch_size : {param_config.n_batch}")
param_config.log.info(f"epoch_size : {param_config.n_epoch}")
param_config.log.info(f"noise_level : {param_config.noise_level}")

model: torch.nn.Module = None

for i in torch.arange(len(param_config.dataset_list)):
    # load dataset
    dataset = param_config.get_dataset_mat(int(i))
    dataset_name = param_config.dataset_list[i]

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = nn.CrossEntropyLoss()
    else:
        param_config.log.war(f"=====Mispara_consq_bias_rsion: Regression=======")
        param_config.loss_fun = RMSELoss()

    # yq_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
    # yq_test_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
    yq_train_acc_tsr = torch.empty(param_config.n_epoch, 0)
    yq_test_acc_tsr = torch.empty(param_config.n_epoch, 0)
    yq_train_kappa_tsr = torch.empty(param_config.n_epoch, 0)
    yq_test_kappa_tsr = torch.empty(param_config.n_epoch, 0)
    yq_train_f1_tsr = torch.empty(param_config.n_epoch, 0)
    yq_test_f1_tsr = torch.empty(param_config.n_epoch, 0)

    for kfold_idx in torch.arange(param_config.n_kfolds):
        param_config.log.war(f"=====k_fold: {kfold_idx + 1}=======")
        train_data, test_data = dataset.get_kfold_data(kfold_idx)
        # # use PCA
        # dim_r = 200
        # estimator = PCA(n_components=dim_r)
        # pca_data_train = estimator.fit_transform(train_data.fea.cpu().numpy())
        # pca_data_test = estimator.transform(test_data.fea.cpu().numpy())
        # train_data.fea = torch.tensor(pca_data_train).to(param_config.device)
        # test_data.fea = torch.tensor(pca_data_test).to(param_config.device)
        # train_data.n_fea = dim_r
        # test_data.n_fea = dim_r

        n_cls = train_data.gnd.unique().shape[0]

        if args.model == "eegnet":
            samplingRate = 225
            Chans = test_data.fea.shape[1]
            n_t = test_data.fea.shape[2]
            dropoutRate = 0.1
            # kernLength = 64
            D = 2
            kernelLength = (int)(samplingRate / 2)  #
            F1 = 2 * Chans  # Double to the number of Channels
            F2 = 4 * F1  # Double to the EEGNet_F1
            # train_data.fea = train_data.fea.view(train_data.n_smpl, Chans, n_t)
            # test_data.fea = test_data.fea.view(test_data.n_smpl, Chans, n_t)
            model = EEGNet(n_cls, Chans, dropoutRate, kernelLength, n_t, F1,  D, F2)
        elif args.model == "fnn_rdm":
            train_data.fea = train_data.fea.view(train_data.n_smpl, -1)
            test_data.fea = test_data.fea.view(test_data.n_smpl, -1)
            model = FnnNormFCRdm(param_config.n_rules, train_data.n_fea, n_cls, param_config.device)
        elif args.model == "fnn_init1":
            print("l")
            # model = FnnNormFCIni(center_arr, std_arr, n_cls, param_config.device)
        elif args.model == "resnet18":
            model = resnet18(num_classes=n_cls)
        elif args.model == "vit":
            Chans = 199
            n_t = 200
            # train_data.fea = train_data.fea.view(train_data.n_smpl, Chans, n_t)
            # test_data.fea = test_data.fea.view(test_data.n_smpl, Chans, n_t)
            fea_expand_train = torch.zeros(train_data.n_smpl, 1, n_t).to(param_config.device)
            train_data.fea = torch.cat([train_data.fea, fea_expand_train], 1)
            fea_expand_test = torch.zeros(test_data.n_smpl, 1, n_t).to(param_config.device)
            test_data.fea = torch.cat([test_data.fea, fea_expand_test], 1)
            image_size = n_t
            patch_size = 40
            num_classes = n_cls
            dim = 128
            depth = 3
            heads = 8
            mlp_dim= 128
            model = ViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
        # # shuffle the samples
        # shuffle_idx = torch.randperm(train_data.n_smpl)
        # train_data.fea = train_data.fea[shuffle_idx, :]
        # train_data.gnd = train_data.gnd[shuffle_idx, :]

        yq_train_acc, yq_train_kappa, yq_train_f1, yq_valid_acc, yq_valid_kappa, yq_valid_f1 = \
            run_model_cls(param_config, train_data, test_data, model)

        yq_test_acc_tsr = torch.cat([yq_test_acc_tsr, yq_valid_acc], 1)
        yq_train_acc_tsr = torch.cat([yq_train_acc_tsr, yq_train_acc], 1)
        yq_test_kappa_tsr = torch.cat([yq_test_kappa_tsr, yq_valid_kappa], 1)
        yq_train_kappa_tsr = torch.cat([yq_train_kappa_tsr, yq_train_kappa], 1)
        yq_test_f1_tsr = torch.cat([yq_test_f1_tsr, yq_valid_f1], 1)
        yq_train_f1_tsr = torch.cat([yq_train_f1_tsr, yq_train_f1], 1)
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    save_dict = dict()
    # save_dict["yq_test_acc_tsr"] = yq_test_acc_tsr.cpu().numpy()
    # save_dict["yq_train_acc_tsr"] = yq_train_acc_tsr.cpu().numpy()
    save_dict["yq_test_acc_tsr"] = yq_test_acc_tsr.numpy()
    save_dict["yq_test_acc_mean_tsr"] = yq_test_acc_tsr.mean(1).numpy()
    save_dict["yq_test_acc_std_tsr"] = yq_test_acc_tsr.std(1).numpy()
    save_dict["yq_train_acc_tsr"] = yq_train_acc_tsr.numpy()
    save_dict["yq_train_acc_mean_tsr"] = yq_train_acc_tsr.mean(1).numpy()
    save_dict["yq_train_acc_std_tsr"] = yq_train_acc_tsr.std(1).numpy()

    save_dict["yq_test_kappa_tsr"] = yq_test_kappa_tsr.numpy()
    save_dict["yq_test_kappa_mean_tsr"] = yq_test_kappa_tsr.mean(1).numpy()
    save_dict["yq_test_kappa_std_tsr"] = yq_test_kappa_tsr.std(1).numpy()
    save_dict["yq_train_kappa_tsr"] = yq_train_kappa_tsr.numpy()
    save_dict["yq_train_kappa_mean_tsr"] = yq_train_kappa_tsr.mean(1).numpy()
    save_dict["yq_train_kappa_std_tsr"] = yq_train_kappa_tsr.std(1).numpy()

    save_dict["yq_test_f1_tsr"] = yq_test_f1_tsr.numpy()
    save_dict["yq_test_f1_mean_tsr"] = yq_test_f1_tsr.mean(1).numpy()
    save_dict["yq_test_f1_std_tsr"] = yq_test_f1_tsr.std(1).numpy()
    save_dict["yq_train_f1_tsr"] = yq_train_f1_tsr.numpy()
    save_dict["yq_train_f1_mean_tsr"] = yq_train_f1_tsr.mean(1).numpy()
    save_dict["yq_train_f1_std_tsr"] = yq_train_f1_tsr.std(1).numpy()
    # save_dict["yq_test_kappa_tsr"] = yq_test_kappa_tsr.numpy()
    # save_dict["yq_train_kappa_tsr"] = yq_train_kappa_tsr.numpy()
    # save_dict["yq_test_f1_tsr"] = yq_test_f1_tsr.numpy()
    # save_dict["yq_train_f1_tsr"] = yq_train_f1_tsr.numpy()


    data_save_file = f"{data_save_dir}/metric_{dataset_name}" \
                     f"_{args.model}_epoch_{param_config.n_epoch}.mat"
    io.savemat(data_save_file, save_dict)
