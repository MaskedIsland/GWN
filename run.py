import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import warnings

from lib.utils import load_adj
from train import Exp_GWN

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default=True, type=eval)
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--devices',type=str,default='cuda:0',help='')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--use_multi_gpu', default=False, type=eval)


    # data
    parser.add_argument('--data', default='PeMSD4', type=str)
    parser.add_argument('--freq', default='d', type=str)
    parser.add_argument('--checkpoints', default='./informer_checkpoints', type=str)
    parser.add_argument('--seq_len', default=12, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    # model
    parser.add_argument('--model', default='GWN', type=str)
    parser.add_argument('--adjdata', default='/content/drive/MyDrive/对比实验/datax/sensor_graph/adj_mx.pkl', type=str)
    parser.add_argument('--adjtype', default='doubletransition', type=str)
    parser.add_argument('--gcn_bool', default=True, type=eval)
    parser.add_argument('--aptonly', default=True, type=eval)
    parser.add_argument('--addaptadj', default=True, type=eval)
    parser.add_argument('--randomadj', default=True, type=eval)
    parser.add_argument('--nhid', default=32, type=int)
    parser.add_argument('--in_dim', default=6, type=int)
    parser.add_argument('--num_nodes', default=307, type=int)

    #train
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--dropout',type=float,default=0.3)
    parser.add_argument('--weight_decay',type=float,default=0.0001)
    parser.add_argument('--print_every',type=int,default=50)
    parser.add_argument('--save',type=str,default='/content/drive/MyDrive/对比实验/model_save_0_1/GWN/PeMSD4-TFDF')
    parser.add_argument('--expid',type=int,default=1)
    # parser.add_argument('--seed', default=10, type=int)
    
    args = parser.parse_args()

    # init_seed(args.seed)

    # device = torch.device(args.device)

    torch.set_default_tensor_type(torch.DoubleTensor)
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
    device = torch.device(args.device)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    if args.aptonly:
        supports = None

    args.adjinit = adjinit
    args.supports = supports
    Exp = Exp_GWN
    exp = Exp(args)
    his_loss = exp.train()
    print("his_loss:", his_loss)
    exp.test(his_loss)