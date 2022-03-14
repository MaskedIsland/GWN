import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

from model.models import gwnet
from lib.dataset import Dataset_PeMSD4
from torch.utils.data import DataLoader
from lib.utils import masked_mae, masked_mape, masked_rmse, loss_weight, drop_feature, cl_loss, metric

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class Exp_GWN(Exp_Basic):
    def __init__(self, args):
        super(Exp_GWN, self).__init__(args)
    
    def _build_model(self):  
        model = gwnet(device = self.args.device, num_nodes = self.args.num_nodes, dropout = self.args.dropout, 
                      supports = self.args.supports, gcn_bool = self.args.gcn_bool, addaptadj = self.args.addaptadj, 
                      aptinit = self.args.aptinit, in_dim = self.args.in_dim, out_dim = self.args.seq_len, 
                      residual_channels = self.args.nhid, dilation_channels = self.args.nhid, skip_channels= self.args.nhid * 8, 
                      end_channels = self.args.nhid * 16)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'PeMSD4':[Dataset_PeMSD4, '/content/drive/MyDrive/对比实验/datax/PEMSD4', 'pems04.npz', self.args.batch_size, True]
        }
        Data = data_dict[self.args.data][0]
        data_root_path = data_dict[self.args.data][1]
        data_file_name = data_dict[self.args.data][2]
        test_batch_size = data_dict[self.args.data][3]
        self.inverse_flag = data_dict[self.args.data][4]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'val':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='test':
            shuffle_flag = False; drop_last = False; batch_size = test_batch_size; freq=args.freq
            # Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            data_root_path = data_root_path,
            data_file_name = data_file_name,
            flag = flag,
            seq_len = self.args.seq_len,
            pred_len = self.args.pred_len,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay = self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion = masked_mae
        return criterion

    def train(self):
        clip = 5
        device = torch.device(self.args.device)
        
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        his_loss =[]
        val_time = []
        train_time = []

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            train_loss = []
            train_mape = []
            train_rmse = []
            predx_loss = []
            clx_loss = []
            t1 = time.time()
            alpha, beta = loss_weight(epoch)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                #train
                model_optim.zero_grad()
                train_x = torch.Tensor(batch_x).to(device)
                train_x = train_x.transpose(1, 3)
                train_y = torch.Tensor(batch_y).to(device)
                train_y = train_y.transpose(1, 3)
                real_val = train_y[:,0,:,:]

                batch_x_aug = drop_feature(train_x.transpose(1, 3), 0.05)
                batch_x_aug2 = drop_feature(train_x.transpose(1, 3), 0.05)
                train_x_aug = batch_x_aug.transpose(1, 3)
                train_x_aug2 = batch_x_aug2.transpose(1, 3)


                input_x = nn.functional.pad(train_x, (1,0,0,0))
                input_x_aug = nn.functional.pad(train_x_aug, (1,0,0,0))
                input_x_aug2 = nn.functional.pad(train_x_aug2, (1,0,0,0))
                output_x, _ = self.model(input_x)
                _, neg_node_emb = self.model(input_x_aug)
                _, neg_node_emb2 = self.model(input_x_aug2)
                loss_cl = cl_loss(neg_node_emb, neg_node_emb2)
                predict = output_x.transpose(1,3)
                real = torch.unsqueeze(real_val,dim=1)
                predict = train_data.inverse_transform(predict, self.inverse_flag)
                real = train_data.inverse_transform(real, self.inverse_flag)
                real = F.relu(real)
                loss_pred = criterion(predict, real, 0.0)
                loss = alpha*loss_pred + beta*loss_cl
                loss.backward()
                model_optim.step()

                mape = masked_mape(predict,real,0.0).item()
                rmse = masked_rmse(predict,real,0.0).item()

                train_loss.append(loss.item())
                predx_loss.append(loss_pred.item())
                clx_loss.append(loss_cl.item())
                train_mape.append(mape)
                train_rmse.append(rmse)

                if i % self.args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Pred Loss: {:.4f}, CL Loss: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(i, train_loss[-1], predx_loss[-1], clx_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            t2 = time.time()
            train_time.append(t2-t1)

            #validation
            s1 = time.time()
            valid_loss, valid_mape, valid_rmse = self.vali(vali_data, vali_loader, criterion)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(epoch,(s2-s1)))
            val_time.append(s2-s1)

            mtrain_loss = np.mean(train_loss)
            mpred_loss = np.mean(predx_loss)
            mcl_loss = np.mean(clx_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)
            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)

            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Pred Loss: {:.4f}, CL Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(epoch, mtrain_loss, mpred_loss, mcl_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            torch.save(self.model.state_dict(), self.args.save+"_epoch_"+str(epoch)+"_"+str(round(mvalid_loss,2))+".pth")

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        return his_loss

    def vali(self, vali_data, vali_loader, criterion):
        device = torch.device(self.args.device)
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            vali_x = torch.Tensor(batch_x).to(device)
            vali_x = vali_x.transpose(1, 3)
            vali_y = torch.Tensor(batch_y).to(device)
            vali_y = vali_y.transpose(1, 3)
            real_val = vali_y[:,0,:,:]

            input_x = nn.functional.pad(vali_x, (1,0,0,0))
            output_x,_ = self.model(input_x)
            predict = output_x.transpose(1, 3)
            real = torch.unsqueeze(real_val,dim=1)
            predict = vali_data.inverse_transform(predict,self.inverse_flag)
            real = vali_data.inverse_transform(real,self.inverse_flag)
            real = F.relu(real)
            loss = criterion(predict, real, 0.0)
            mape = masked_mape(predict,real,0.0).item()
            rmse = masked_rmse(predict,real,0.0).item()

            valid_loss.append(loss.item())
            valid_mape.append(mape)
            valid_rmse.append(rmse)
        return valid_loss, valid_mape, valid_rmse
    
    def test(self, his_loss):
        device = torch.device(self.args.device)
        bestid = np.argmin(his_loss)
        self.model.load_state_dict(torch.load(self.args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
        test_data, test_loader = self._get_data(flag = 'test')
        self.model.eval()
        outputs = []
        realy = []
        for iter, (x, y) in enumerate(test_loader):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            testy = torch.Tensor(y).to(device)
            with torch.no_grad():
                preds,_ = self.model(testx)
            preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())
            realy.append(testy)
        realy = torch.cat(realy, dim=0)
        yreal = realy.transpose(1,3)[:,0,:,:]
        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]
        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))

        amae = []
        amape = []
        armse = []
        for i in range(12):
            pred = test_data.inverse_transform(yhat[:,:,i],self.inverse_flag).to(device)
            real = test_data.inverse_transform(yreal[:,:,i],self.inverse_flag).to(device)
            real = F.relu(real)
            metrics = metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(self.model.state_dict(), self.args.save+"_exp"+str(self.args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
 