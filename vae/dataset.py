import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb
import numpy as np
import h5py
from torch.utils.data import RandomSampler


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class SurfJDataset(Dataset):

    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.sample_folder = root + '/data/'
        # load file ids
        total_wo_test_ids = root +'/ids/total_wo_test_ids.hdf5'
        self.total_wo_test_ids =  self.read_hdf5(total_wo_test_ids, 'total_wo_test_ids')

        rela_pass_wd_ids = root +'/ids/rela_pass_wd_ids.hdf5'
        self.rela_pass_wd_ids =  self.read_hdf5(rela_pass_wd_ids, 'rela_pass_wd_ids')

        rela_sd_wd_ids = root +'/ids/rela_sd_wd_ids.hdf5'
        self.rela_sd_wd_ids =  self.read_hdf5(rela_sd_wd_ids, 'rela_sd_wd_ids')

        rela_stop_wd_ids = root +'/ids/rela_stop_wd_ids.hdf5'
        self.rela_stop_wd_ids =  self.read_hdf5(rela_stop_wd_ids, 'rela_stop_wd_ids')

        to_clned_ids = root +'/ids/to_clned_ids.hdf5'
        self.to_clned_ids =  self.read_hdf5(to_clned_ids, 'to_clned_ids')

        total_ids = root +'/ids/total_ids.hdf5'
        self.total_ids =  self.read_hdf5(total_ids, 'total_ids')
        # # collections of the folders
        # self.sample_folder = root + '/data/'
        # self.subfolders = [f.name for f in os.scandir(self.sample_folder) if f.is_dir()]
        # frame range
        frame_ranges_file = root +'/info/j_range.csv'
        self.frame_ranges = torch.tensor(np.genfromtxt(frame_ranges_file, delimiter=','))
        self.maxj = torch.max(abs(self.frame_ranges))
        # s range
        s_ranges_file = root +'/info/s_range.csv'
        self.s_ranges = torch.tensor(np.genfromtxt(s_ranges_file, delimiter=','))
        self.mins = self.s_ranges[0]
        self.maxs = self.s_ranges[1]

    def read_hdf5(self, file, data_name):
        with h5py.File(file, "r") as f:
            content = f[data_name][:]
        return content

    def __len__(self) -> int:
        return len(self.total_wo_test_ids)

    def __getitem__(self, index: int):
        current_subfolders = self.total_wo_test_ids[index].decode('utf-8')
        current_path = self.sample_folder + current_subfolders
        current_surfJ = current_path + '/surfj.hdf5'
        self.surfJ =  self.read_hdf5(current_surfJ, 'surfj')
        cur_surfJ = np.squeeze(self.surfJ)
        cur_surfJ = torch.tensor(cur_surfJ)
        # batch_img = Image.fromarray(cur_surfJ).convert('L')
        # pipeline = torch.tensor(arr2)
        # calc log magnitudes of x-oriented and y-oriented currents
        scale = torch.log(self.maxj)
        x_r_currents = cur_surfJ[0:1,:,:,:]
        x_i_currents = cur_surfJ[1:2,:,:,:]
        y_r_currents = cur_surfJ[2:3,:,:,:]
        y_i_currents = cur_surfJ[3:4,:,:,:]
        x_mag_currents = torch.sqrt((torch.pow(x_r_currents,2)+torch.pow(x_i_currents,2)))
        y_mag_currents = torch.sqrt((torch.pow(y_r_currents,2)+torch.pow(y_i_currents,2)))
        x_logmag_currents = torch.log(x_mag_currents+1)
        y_logmag_currents = torch.log(y_mag_currents+1)
        logmag_currents = torch.cat((x_logmag_currents,y_logmag_currents),dim=0)
        # log magnitude
        pos_cur_surfJ = torch.where(cur_surfJ>=0, cur_surfJ, 0)
        neg_cur_surfJ = torch.where(cur_surfJ<0, cur_surfJ, 0)
        logpos_cur_surfJ = torch.log(pos_cur_surfJ+1)
        logneg_cur_surfJ = -torch.log(-neg_cur_surfJ+1)
        log_cur_surfJ = logpos_cur_surfJ+logneg_cur_surfJ
        # rescale
        scaledlog_cur_surfJ = log_cur_surfJ/scale
        scaledlogmag_currents = logmag_currents/scale
        scaledlog_cur_surfJ = scaledlog_cur_surfJ.permute(1,0,2,3)
        scaledlogmag_currents =  scaledlogmag_currents.permute(1,0,2,3)

        # load topo
        current_topo = current_path + '/topo.hdf5'
        self.topo =  self.read_hdf5(current_topo, 'topo')
        cur_topo = np.squeeze(self.topo)
        cur_topo = torch.tensor(cur_topo)
        cur_topo = cur_topo.unsqueeze(0).unsqueeze(0)
        b = scaledlog_cur_surfJ.shape[0]
        c = scaledlog_cur_surfJ.shape[1]
        d = scaledlogmag_currents.shape[1]
        curJ_topo = cur_topo.repeat(b,c,1,1)
        curMag_topo = cur_topo.repeat(b,d,1,1)
        scaledlog_cur_surfJ[:,:,:,:][curJ_topo[:,:,:,:] <= 100.] = 0.
        scaledlogmag_currents[:,:,:,:][curMag_topo[:,:,:,:] <= 100.] = 0.
        temptopo = torch.tensor(self.topo)
        scaledtopo = torch.ones(temptopo.shape)
        scaledtopo[temptopo <= 100.] = 0.

        # load s parameters
        current_s = current_path + '/s.hdf5'
        self.s=  self.read_hdf5(current_s, 's')
        self.s = torch.tensor(self.s)
        # normalize s to (0,1)
        s_normalize = (self.s-self.mins)/(self.maxs-self.mins)
        
        # add freq ids
        f_ids = torch.arange(0.2,2.1,0.1)
        s_normalize = s_normalize.unsqueeze(-1)
        f_ids = f_ids.unsqueeze(-1)
        s_normalize = torch.cat((s_normalize,f_ids),dim=-1)

        # ramdomly choose 10 frequencies from 19
        f_range = range(0,19)
        f_list = list(f_range)
        f_sample = RandomSampler(f_list)
        f_random = [i for i in f_sample][0:19]
        f_random.sort()
        f_random = torch.tensor(f_random)

        # return corresponding frames of data according to f_random
        scaledlog_cur_surfJ = scaledlog_cur_surfJ[f_random,:,:,:]
        scaledlogmag_currents = scaledlogmag_currents[f_random,:,:,:]
        s_normalize = s_normalize[f_random,:]

        # data enhancement by symetrical properties
        if ~(self.total_wo_test_ids[index] in self.to_clned_ids):
            if torch.rand(1)[0]>= 0.5:
                scaledlogmag_currents = np.roll(scaledlogmag_currents,int(scaledlogmag_currents.shape[2]/2),2)
                scaledlogmag_currents = np.roll(scaledlogmag_currents,int(scaledlogmag_currents.shape[3]/2),3)
                scaledlog_cur_surfJ = np.roll(scaledlog_cur_surfJ,int(scaledlog_cur_surfJ.shape[2]/2),2)
                scaledlog_cur_surfJ = np.roll(scaledlog_cur_surfJ,int(scaledlog_cur_surfJ.shape[3]/2),3)

        scaledlog_cur_surfJ = torch.tensor(scaledlog_cur_surfJ)
        scaledlogmag_currents = torch.tensor(scaledlogmag_currents)
        return scaledlog_cur_surfJ.float(), scaledlogmag_currents.float(), s_normalize.float(), scaledtopo.float(), f_random, current_subfolders

class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename

class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename
