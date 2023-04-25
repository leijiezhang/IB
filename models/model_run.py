import torch
import sklearn.metrics as mtxc
from utils.param_config import ParamConfig
from utils.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from utils.loss_utils import HLoss
import math
from kmeans_pytorch import kmeans
from utils.loss_utils import NRMSELoss, LikelyLoss, LossFunc, MSELoss
from models.fpn_models import *
from utils.model_utils import mixup_data
from utils.dataset import DatasetTorch, DatasetTorchB
import scipy.io as io
from models.h_utils import HNormal
from models.rules import RuleKmeans
from models.fnn_solver import FnnSolveReg


def fnn_cls(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param test_data: test dataset
        :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_acc = LikelyLoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_acc = LikelyLoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training acc of traditional FNN: {fnn_train_acc}")
    param_config.log.info(f"Test acc of test traditional FNN: {fnn_test_acc}")
    return fnn_train_acc, fnn_test_acc


def fnn_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param test_data: test dataset
        :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = NRMSELoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = NRMSELoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training me of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test mse of test traditional FNN: {fnn_test_mse}")
    return fnn_train_mse, fnn_test_mse


def fpn_cls(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FpnMlpFsCls_1(prototype_list, std, n_cls, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = torch.empty(0, 1).to(param_config.device)
    fpn_valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num.float() / gnd_train.shape[0]
            fpn_train_acc = torch.cat([fpn_train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num / gnd_val.shape[0]
            fpn_valid_acc = torch.cat([fpn_valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        idx = fpn_model.fire_strength.max(1)[1]
        idx_unique = idx.unique(sorted=True)
        idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < acc_train:
        #     best_test_rslt = acc_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train acc : {fpn_train_acc[-1, 0]}, test acc : {fpn_valid_acc[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_acc, fpn_valid_acc


def fpn_reg(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FpnMlpFsCls_1(prototype_list, std, n_cls, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    fpn_train_mse = torch.empty(0, 1).to(param_config.device)
    fpn_valid_mse = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            mse_train = MSELoss().forward(outputs_train, gnd_train)
            fpn_train_mse = torch.cat([fpn_train_mse, mse_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            mse_val = MSELoss().forward(outputs_val, gnd_val)
            fpn_valid_mse = torch.cat([fpn_valid_mse, mse_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # if best_test_rslt < mse_train:
        #     best_test_rslt = mse_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train mse : {fpn_train_mse[-1, 0]}, test mse : {fpn_valid_mse[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_mse, fpn_valid_mse


def fpn_run_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = NRMSELoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = NRMSELoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training me of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test mse of test traditional FNN: {fnn_test_mse}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    # model: nn.Module = MLP(train_data.fea.shape[1])
    # rules = RuleKmeans()
    # rules.fit(train_data.fea, param_config.n_rules)
    prototype_list = rules.center_list
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    fpn_model: nn.Module = FpnMlpFsReg(prototype_list, param_config.device)
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])

    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    fpn_train_losses = []
    fpn_valid_losses = []

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
                      f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
                      f"k_{current_k}.pkl"
    # #load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, 1).to(param_config.device)
        outputs_val = torch.empty(0, 1).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            loss_train = NRMSELoss().forward(outputs_train, gnd_train)
            fpn_train_losses.append(loss_train.item())
            for i, (data, labels) in enumerate(valid_loader):
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            loss_val = NRMSELoss().forward(outputs_val, gnd_val)
            fpn_valid_losses.append(loss_val.item())
        param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        if best_test_rslt < loss_train:
            torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train loss : {fpn_train_losses[-1]}, test loss : {fpn_valid_losses[-1]}")

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_losses"] = torch.tensor(fpn_train_losses).numpy()
    save_dict["fpn_valid_losses"] = torch.tensor(fpn_valid_losses).numpy()
    save_dict["fnn_train_mse"] = fnn_train_mse.numpy()
    save_dict["fnn_test_mse"] = fnn_test_mse.numpy()
    data_save_file = f"{data_save_dir}/mse_bpfnn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_lr_{param_config.lr:.6f}" \
                     f"_k_{current_k}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_losses, fpn_valid_losses


def run_fnn_fnn_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_acc, fpn_valid_acc = fnn_fnn_mlp(param_config, train_data, train_loader, valid_loader)

    # plt.figure(0)
    # title = f"FPN Acc of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    # plt.title(title)
    # plt.xlabel('Epoch')
    # plt.ylabel('Acc')
    # plt.plot(torch.arange(len(fpn_valid_acc)), fpn_train_acc.cpu(), 'b-', linewidth=2,
    #          markersize=5)
    # plt.plot(torch.arange(len(fpn_valid_acc)), fpn_valid_acc.cpu(), 'r--', linewidth=2,
    #          markersize=5)
    # plt.legend(['fpn train', 'fpn test'])
    # plt.savefig(f"{data_save_dir}/acc_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
    #             f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_acc, fpn_valid_acc


def fnn_fnn_mlp(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = torch.empty(0, 1).to(param_config.device)
    fpn_valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num.float() / gnd_train.shape[0]
            fpn_train_acc = torch.cat([fpn_train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num / gnd_val.shape[0]
            fpn_valid_acc = torch.cat([fpn_valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < acc_train:
        #     best_test_rslt = acc_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train acc : {fpn_train_acc[-1, 0]}, test acc : {fpn_valid_acc[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_acc, fpn_valid_acc


def run_fnn_fnn_mlp_r(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_acc, fpn_valid_acc = fnn_fnn_mlp_r(param_config, train_data, train_loader, valid_loader)

    return fpn_train_acc, fpn_valid_acc


def fnn_fnn_mlp_r(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    n_cls = 1
    fpn_model: nn.Module = FnnNormFCIni(prototype_list, std, n_cls, param_config.device)
    # fpn_model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = torch.empty(0, 1).to(param_config.device)
    fpn_valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            acc_train = loss_fn(outputs_train, gnd_train)
            fpn_train_acc = torch.cat([fpn_train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            acc_val = loss_fn(outputs_val, gnd_val)
            fpn_valid_acc = torch.cat([fpn_valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < acc_train:
        #     best_test_rslt = acc_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train mse : {fpn_train_acc[-1, 0]}, test mse : {fpn_valid_acc[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_acc, fpn_valid_acc


def run_fnn_fnn_mlp_rdm_r(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_acc, fpn_valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return fpn_train_acc, fpn_valid_acc


def fnn_fnn_mlp_rdm_r(param_config: ParamConfig, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    # prototype_ids, prototype_list = kmeans(
    #     X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
    #     device=torch.device(train_data.fea.device)
    # )
    # prototype_list = prototype_list.to(param_config.device)
    # # get the std of data x
    # std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    # for i in range(param_config.n_rules):
    #     mask = prototype_ids == i
    #     cluster_samples = train_data.fea[mask]
    #     std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
    #         cluster_samples.shape[0]).float())
    #     # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
    #     std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    # std = torch.where(std < 10 ** -5,
    #                   10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    n_cls = 1
    fpn_model: nn.Module = FnnMlpMlpRdm(param_config.n_rules, train_loader.dataset.x.shape[1], n_cls, 
                                        param_config.drop_rate, param_config.device)
    # fpn_model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = torch.empty(0, 1).to(param_config.device)
    fpn_valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            acc_train = loss_fn(outputs_train, gnd_train)
            fpn_train_acc = torch.cat([fpn_train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            acc_val = loss_fn(outputs_val, gnd_val)
            fpn_valid_acc = torch.cat([fpn_valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < acc_train:
        #     best_test_rslt = acc_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train mse : {fpn_train_acc[-1, 0]}, test mse : {fpn_valid_acc[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_acc, fpn_valid_acc


def run_model(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, model: nn.Module):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param model: models k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============run models===========
    n_cls = 1
    model: nn.Module = model.to(param_config.device)
    # model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    train_acc = torch.empty(0, 1).to(param_config.device)
    valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    for epoch in range(epochs):
        model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = model(data)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            acc_train = loss_fn(outputs_train, gnd_train)
            train_acc = torch.cat([train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            acc_val = loss_fn(outputs_val, gnd_val)
            valid_acc = torch.cat([valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train mse : {train_acc[-1, 0]}, test mse : {valid_acc[-1, 0]}")

    param_config.log.info("training epoch:=======================finished===========================")
    # train_acc, valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return train_acc, valid_acc


def run_model_cls_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, model: nn.Module, arrange_arr):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param model: models k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============run models===========
    model: nn.Module = model.to(param_config.device)
    # model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_config.lr)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    train_acc = torch.empty(0, 1).to(param_config.device)
    valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    for epoch in range(epochs):
        model.train()

        for i, (_, data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            # data, labels = mixup_data(data, labels, alpha=1.0, mixup_rate=1)
            outputs_temp = model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        index_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        index_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (index, data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                index = index.to(param_config.device)
                outputs_temp = model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                index_train = torch.cat((index_train, index.unsqueeze(1)), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num.float() / gnd_train.shape[0]
            loss_train = loss_fn(outputs_train, gnd_train.squeeze().long())
            # nn.Softmax()(outputs_train)
            ang_orig_train = train_data.gnd_orig[index_train.squeeze().long()]
            ang_train_v1 = (nn.Softmax()(outputs_train)*arrange_arr[1::]).sum(1)
            loss_train_v1 = nn.L1Loss()(ang_train_v1, ang_orig_train.squeeze())
            ang_train_v2 = (nn.Softmax()(outputs_train) * (arrange_arr[1::]-(arrange_arr[1]-arrange_arr[0])/2)).sum(1)
            loss_train_v2 = nn.L1Loss()(ang_train_v2, ang_orig_train.squeeze())
            ang_train_v3 = arrange_arr[1::][predicted_train]
            loss_train_v3 = nn.L1Loss()(ang_train_v3, ang_orig_train.squeeze())
            ang_train_v4 = (arrange_arr[1::]-(arrange_arr[1]-arrange_arr[0])/2)[predicted_train]
            loss_train_v4 = nn.L1Loss()(ang_train_v4, ang_orig_train.squeeze())
            
            # train_acc = torch.cat([train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (index, data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                index = index.to(param_config.device)
                outputs_temp = model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                index_val = torch.cat((index_val, index.unsqueeze(1)), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            # acc_val = loss_fn(outputs_val, gnd_val)
            # valid_acc = torch.cat([valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num.float() / gnd_val.shape[0]
            loss_val = loss_fn(outputs_val, gnd_val.squeeze().long())

            ang_orig_val = test_data.gnd_orig[index_val.squeeze().long()]
            ang_val_v1 = (nn.Softmax()(outputs_val) * arrange_arr[1::]).sum(1)
            loss_val_v1 = nn.L1Loss()(ang_val_v1, ang_orig_val.squeeze())
            ang_val_v2 = (
                        nn.Softmax()(outputs_val) * (arrange_arr[1::] - (arrange_arr[1] - arrange_arr[0]) / 2)).sum(1)
            loss_val_v2 = nn.L1Loss()(ang_val_v2, ang_orig_val).squeeze()
            ang_val_v3 = arrange_arr[1::][predicted_val]
            loss_val_v3 = nn.L1Loss()(ang_val_v3, ang_orig_val.squeeze())
            ang_val_v4 = (arrange_arr[1::] - (arrange_arr[1] - arrange_arr[0]) / 2)[predicted_val]
            loss_val_v4 = nn.L1Loss()(ang_val_v4, ang_orig_val.squeeze())
        param_config.log.war(
            f"epoch : {epoch + 1}, train_acc: {acc_train:.4f}, train_Loss: {loss_train:.4f} train mae : {loss_train_v1:.4f},{loss_train_v2:.4f}, "
            f"{loss_train_v3:.4f},{loss_train_v4:.4f},\n "
            f"                                      val_acc: {acc_val:.4f},  val_Loss: {loss_val:.4f} test mse : {loss_val_v1:.4f},"
            f"{loss_val_v2:.4f}, {loss_val_v3:.4f},{loss_val_v4:.4f}")

    param_config.log.info("training epoch:=======================finished===========================")
    # train_acc, valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return train_acc, valid_acc


def run_model_cls(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, model: nn.Module):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param model: models k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============run models===========
    model: nn.Module = model.to(param_config.device)
    # model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_config.lr)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    # train_acc_list = torch.empty(0, 1).to(param_config.device)
    # valid_acc_list = torch.empty(0, 1).to(param_config.device)
    train_acc_list = torch.empty(0, 1)
    valid_acc_list = torch.empty(0, 1)
    train_kappa_list = torch.empty(0, 1)
    valid_kappa_list = torch.empty(0, 1)
    train_f1_list = torch.empty(0, 1)
    valid_f1_list = torch.empty(0, 1)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    for epoch in range(epochs):
        model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            # data, labels = mixup_data(data, labels, alpha=1.0, mixup_rate=1)
            outputs_temp = model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        # index_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        # index_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                # index = index.to(param_config.device)
                outputs_temp = model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                # index_train = torch.cat((index_train, index.unsqueeze(1)), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            predicted_train_cpu_nm = predicted_train.cpu().numpy()
            gnd_train_cpu_nm = gnd_train.cpu().numpy()
            acc_train = torch.tensor(mtxc.accuracy_score(gnd_train_cpu_nm, predicted_train_cpu_nm)).unsqueeze(0).unsqueeze(0)
            kappa_train = torch.tensor(mtxc.cohen_kappa_score(gnd_train_cpu_nm, predicted_train_cpu_nm)).unsqueeze(0).unsqueeze(0)
            f1_train = torch.tensor(mtxc.cohen_kappa_score(gnd_train_cpu_nm, predicted_train_cpu_nm)).unsqueeze(0).unsqueeze(0)
            train_acc_list = torch.cat([train_acc_list, acc_train], 0)
            train_kappa_list = torch.cat([train_kappa_list, kappa_train], 0)
            train_f1_list = torch.cat([train_f1_list, f1_train], 0)
            # correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            # acc_train = correct_train_num.float() / gnd_train.shape[0]
            # train_acc_list = torch.cat([train_acc_list, acc_train.unsqueeze(0).unsqueeze(0)], 0)
            loss_train = loss_fn(outputs_train, gnd_train.squeeze().long())
            # nn.Softmax()(outputs_train)

            # train_acc = torch.cat([train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                # index = index.to(param_config.device)
                outputs_temp = model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                # index_val = torch.cat((index_val, index.unsqueeze(1)), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            # acc_val = loss_fn(outputs_val, gnd_val)
            # valid_acc = torch.cat([valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)
            _, predicted_val = torch.max(outputs_val, 1)
            predicted_val_cpu_nm = predicted_val.cpu().numpy()
            gnd_val_cpu_nm = gnd_val.cpu().numpy()
            acc_val = torch.tensor(mtxc.accuracy_score(gnd_val_cpu_nm, predicted_val_cpu_nm)).unsqueeze(
                0).unsqueeze(0)
            kappa_val = torch.tensor(mtxc.cohen_kappa_score(gnd_val_cpu_nm, predicted_val_cpu_nm)).unsqueeze(
                0).unsqueeze(0)
            f1_val = torch.tensor(mtxc.f1_score(gnd_val_cpu_nm, predicted_val_cpu_nm)).unsqueeze(
                0).unsqueeze(0)
            valid_acc_list = torch.cat([valid_acc_list, acc_val], 0)
            valid_kappa_list = torch.cat([valid_kappa_list, kappa_val], 0)
            valid_f1_list = torch.cat([valid_f1_list, f1_val], 0)
            # correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            # acc_val = correct_val_num.float() / gnd_val.shape[0]
            # valid_acc_list = torch.cat([valid_acc_list, acc_val.unsqueeze(0).unsqueeze(0)], 0)
            loss_val = loss_fn(outputs_val, gnd_val.squeeze().long())

        param_config.log.war(
            f"epoch : {epoch + 1}, train: {float(acc_train):.4f}|{float(kappa_train):.4f}|{float(f1_train):.4f}, train_Loss: {loss_train:.4f} "
            f"val: {float(acc_val):.4f}|{float(kappa_val):.4f}|{float(f1_val):.4f}, val_Loss: {loss_val:.4f}")

    param_config.log.info("training epoch:=======================finished===========================")
    # train_acc, valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return train_acc_list, train_kappa_list, train_f1_list, valid_acc_list, valid_kappa_list, valid_f1_list


def run_fnn_cls(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, model: nn.Module):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param model: models k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd, y_orig=train_data.gnd_orig)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd, y_orig=train_data.gnd_orig)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============run models===========
    model: nn.Module = model.to(param_config.device)
    # model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_config.lr)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    train_acc = torch.empty(0, 1).to(param_config.device)
    valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    for epoch in range(epochs):
        model.train()

        for i, (data, labels, _) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            # data, labels = mixup_data(data, labels, alpha=1.0, mixup_rate=1)
            outputs_temp, fr_temp = model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        # index_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        # index_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels, _) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                # index = index.to(param_config.device)
                outputs_temp, fr_temp = model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                # index_train = torch.cat((index_train, index.unsqueeze(1)), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num.float() / gnd_train.shape[0]
            loss_train = loss_fn(outputs_train, gnd_train.squeeze().long())
            # nn.Softmax()(outputs_train)

            # train_acc = torch.cat([train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels, _) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                # index = index.to(param_config.device)
                outputs_temp, fr_temp = model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                # index_val = torch.cat((index_val, index.unsqueeze(1)), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            # acc_val = loss_fn(outputs_val, gnd_val)
            # valid_acc = torch.cat([valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num.float() / gnd_val.shape[0]
            loss_val = loss_fn(outputs_val, gnd_val.squeeze().long())

        param_config.log.war(
            f"epoch : {epoch + 1}, train_acc: {acc_train:.4f}, train_Loss: {loss_train:.4f} "
            f"val_acc: {acc_val:.4f}, val_Loss: {loss_val:.4f}")

    param_config.log.info("training epoch:=======================finished===========================")
    # train_acc, valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return train_acc, valid_acc


def run_model_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, model: nn.Module, loss_fn):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param model: models k
    :param loss_fn: loss function
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd, y_orig=train_data.gnd_orig)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd, y_orig=train_data.gnd_orig)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)

    # ============run models===========
    model: nn.Module = model.to(param_config.device)
    # model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_config.lr)
    epochs = param_config.n_epoch

    train_acc = torch.empty(0, 1).to(param_config.device)
    valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    loss_entropy_fn = HLoss()
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    for epoch in range(epochs):
        model.train()

        for i, (data, labels, _) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            data, labels = mixup_data(data, labels, alpha=1.0, mixup_rate=1)
            outputs_temp = model(data)
            loss = loss_fn(outputs_temp, labels.squeeze())
            # loss_entropy = loss_entropy_fn(fr_temp)
            # loss = loss_label + 0 * loss_entropy
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_train = torch.empty(0).to(param_config.device)
        outputs_val = torch.empty(0).to(param_config.device)

        gnd_train = torch.empty(0).to(param_config.device)
        gnd_val = torch.empty(0).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels, _) in enumerate(train_loader):
                outputs_temp = model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels.squeeze(1)), 0)
            loss_train = nn.L1Loss()(outputs_train, gnd_train)

            for i, (data, labels, _) in enumerate(valid_loader):
                outputs_temp = model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels.squeeze(1)), 0)
            loss_val = nn.L1Loss()(outputs_val, gnd_val)

        param_config.log.war(
            f"epoch : {epoch + 1}, train_Loss: {loss_train:.4f} "
            f", val_Loss: {loss_val:.4f}/{loss_val*180/math.pi:.4f}")

    param_config.log.info("training epoch:=======================finished===========================")
    # train_acc, valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return train_acc, valid_acc


def run_model_reg_fnn(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, model: nn.Module, loss_fn):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param model: models k
    :param loss_fn: loss function
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd, y_orig=train_data.gnd_orig)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd, y_orig=train_data.gnd_orig)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)

    # ============run models===========
    model: nn.Module = model.to(param_config.device)
    # model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_config.lr)
    epochs = param_config.n_epoch

    train_acc = torch.empty(0, 1).to(param_config.device)
    valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    loss_entropy_fn = HLoss()
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    for epoch in range(epochs):
        model.train()

        for i, (data, labels, _) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            data, labels = mixup_data(data, labels, alpha=1.0, mixup_rate=1)
            outputs_temp, fr_temp = model(data)
            loss_label = loss_fn(outputs_temp, labels.squeeze())
            loss_entropy = loss_entropy_fn(fr_temp)
            loss = loss_label + 0.0 * loss_entropy
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_train = torch.empty(0).to(param_config.device)
        outputs_val = torch.empty(0).to(param_config.device)

        gnd_train = torch.empty(0).to(param_config.device)
        gnd_val = torch.empty(0).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels, _) in enumerate(train_loader):
                outputs_temp, _ = model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels.squeeze(1)), 0)
            loss_train = nn.L1Loss()(outputs_train, gnd_train)

            for i, (data, labels, _) in enumerate(valid_loader):
                outputs_temp, _ = model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels.squeeze(1)), 0)
            loss_val = nn.L1Loss()(outputs_val, gnd_val)

        param_config.log.war(
            f"epoch : {epoch + 1}, train_Loss: {loss_train:.4f} "
            f", val_Loss: {loss_val:.4f}/{loss_val*180/math.pi:.4f}")

    param_config.log.info("training epoch:=======================finished===========================")
    # train_acc, valid_acc = fnn_fnn_mlp_rdm_r(param_config, train_loader, valid_loader)

    return train_acc, valid_acc


def eegnet_reg(param_config: ParamConfig, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    # prototype_ids, prototype_list = kmeans(
    #     X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
    #     device=torch.device(train_data.fea.device)
    # )
    # prototype_list = prototype_list.to(param_config.device)
    # # get the std of data x
    # std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    # for i in range(param_config.n_rules):
    #     mask = prototype_ids == i
    #     cluster_samples = train_data.fea[mask]
    #     std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
    #         cluster_samples.shape[0]).float())
    #     # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
    #     std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    # std = torch.where(std < 10 ** -5,
    #                   10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    n_cls = 1
    fpn_model: nn.Module = EEGNet().to(param_config.device)
    # fpn_model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = torch.empty(0, 1).to(param_config.device)
    fpn_valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            acc_train = loss_fn(outputs_train, gnd_train)
            fpn_train_acc = torch.cat([fpn_train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            # _, predicted_val = torch.max(outputs_val, 1)
            acc_val = loss_fn(outputs_val, gnd_val)
            fpn_valid_acc = torch.cat([fpn_valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < acc_train:
        #     best_test_rslt = acc_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train mse : {fpn_train_acc[-1, 0]}, test mse : {fpn_valid_acc[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_acc, fpn_valid_acc


def run_fnn_fnn_fc(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_acc, fpn_valid_acc = fnn_fnn_fc(param_config, train_data, train_loader, valid_loader)

    # plt.figure(0)
    # title = f"FPN Acc of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    # plt.title(title)
    # plt.xlabel('Epoch')
    # plt.ylabel('Acc')
    # plt.plot(torch.arange(len(fpn_valid_acc)), fpn_train_acc.cpu(), 'b-', linewidth=2,
    #          markersize=5)
    # plt.plot(torch.arange(len(fpn_valid_acc)), fpn_valid_acc.cpu(), 'r--', linewidth=2,
    #          markersize=5)
    # plt.legend(['fpn train', 'fpn test'])
    # plt.savefig(f"{data_save_dir}/acc_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
    #             f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_acc, fpn_valid_acc


def fnn_fnn_fc(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FnnFcFnnFCIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = torch.empty(0, 1).to(param_config.device)
    fpn_valid_acc = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num.float() / gnd_train.shape[0]
            fpn_train_acc = torch.cat([fpn_train_acc, acc_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num / gnd_val.shape[0]
            fpn_valid_acc = torch.cat([fpn_valid_acc, acc_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < acc_train:
        #     best_test_rslt = acc_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train acc : {fpn_train_acc[-1, 0]}, test acc : {fpn_valid_acc[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_acc, fpn_valid_acc


def fpn_run_cls_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_acc, fpn_valid_acc = fpn_cls(param_config, train_data, train_loader, valid_loader)

    # plt.figure(0)
    # title = f"FPN Acc of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    # plt.title(title)
    # plt.xlabel('Epoch')
    # plt.ylabel('Acc')
    # plt.plot(torch.arange(len(fpn_valid_acc)), fpn_train_acc.cpu(), 'b-', linewidth=2,
    #          markersize=5)
    # plt.plot(torch.arange(len(fpn_valid_acc)), fpn_valid_acc.cpu(), 'r--', linewidth=2,
    #          markersize=5)
    # plt.legend(['fpn train', 'fpn test'])
    # plt.savefig(f"{data_save_dir}/acc_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
    #             f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_acc, fpn_valid_acc


def fpn_run_reg_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_mse, fpn_valid_mse = fpn_reg(param_config, train_data, train_loader, valid_loader)

    plt.figure(0)
    title = f"FPN mse of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('mse')
    plt.plot(torch.arange(len(fpn_valid_mse)), fpn_train_mse.cpu(), 'r-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_mse)), fpn_valid_mse.cpu(), 'r--', linewidth=2,
             markersize=5)
    plt.legend(['fpn train', 'fpn test'])
    plt.savefig(f"{data_save_dir}/mse_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
                f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_mse, fpn_valid_mse


