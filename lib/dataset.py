from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os

from lib.FeatureTime import time_features
from utils import StandardScaler2

class Dataset_PeMSD4(Dataset):
    def __init__(self, data_root_path, data_file_name, flag='train', 
                 seq_len = 12, pred_len = 12,  
                 scale=True, inverse=True, timeenc=0, freq='d'):
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.root_path = data_root_path
        self.data_path = data_file_name
        self.__read_data__()

    def __read_data__(self):
        raw_data = np.load(os.path.join(self.root_path, self.data_path),allow_pickle=True)['data'][:,:,0]
        num_samples, num_nodes = raw_data.shape
        train_num = round(num_samples * 0.7)
        val_num = round(num_samples * 0.2)
        test_num = num_samples - train_num - val_num

        border1s = [0, train_num + 1, train_num + val_num + 1]
        border2s = [train_num, train_num + val_num, train_num + val_num + test_num]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        date_list = []
        start_time = '2018-01-01 00:00:00'
        start_time = pd.to_datetime(start_time)
        cur_date = start_time
        for i in range(num_samples):
            date_list.append(cur_date)
            append_date = cur_date + timedelta(minutes=5)
            cur_date = append_date 
        date_arr = np.array(date_list)
        date_arr_ser = pd.Series(date_arr)
        df_dates = pd.DataFrame(date_arr_ser,columns=['date'])
        df_data = np.expand_dims(raw_data, axis=-1)
        feature_list = [df_data]
        df_stamp = time_features(df_dates, 1, 't')
        time_feat_num = df_stamp.shape[1]
        for index in range(time_feat_num):
            time_feat = np.tile(df_stamp[:,index], [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(time_feat)
        df_data = np.concatenate(feature_list, axis=-1)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler2(mean=train_data[..., 0].mean(), std=train_data[..., 0].std())
            df_data[..., 0] = self.scaler.transform(df_data[..., 0])
            data = df_data
        else:
            data = df_data

        self.data = data[border1:border2] 
    
    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end + 1
        y_end = y_begin + self.pred_len
        data_x = self.data[x_begin : x_end]
        data_y = self.data[y_begin : y_end]

        return data_x, data_y
    
    def __len__(self):
        return len(self.data) - self.pred_len - self.pred_len

    def inverse_transform(self, data, inverse_flag=True):
        if inverse_flag:
          return self.scaler.inverse_transform(data)
        else:
          return data