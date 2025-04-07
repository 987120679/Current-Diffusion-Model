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

    def __init__(self, root, num_frames) -> None:
        super().__init__()
        self.root = root
        self.sample_folder = root + '/data/'
        self.hdf5files_folder = root + '/s_data/'
        self.subfolders = [f.name for f in os.scandir(self.sample_folder) if f.is_dir()]
        self.hdf5files = [f.name[:-5] for f in os.scandir(self.hdf5files_folder)]
        self.select1 = ['Prj_id21_t240402_002225','Prj_id34_t240326_163621','Prj_id53_t240318_194450','Prj_id54_t240811_043357','Prj_id377_t240229_133849']
        self.paper1 = ['Prj_id145_t240626_060320']
        self.paper2 = ['Prj_id43_t240706_053641']
        self.num_frames = num_frames
        # load file ids

        total_wo_test_ids = root +'/ids/total_wo_test_ids.hdf5'
        self.total_wo_test_ids =  self.read_hdf5(total_wo_test_ids, 'total_wo_test_ids').tolist()
        
        total_wo_test_ids = root +'/ids/total_wo_test_ids.hdf5'
        self.total_wo_test_ids =  self.read_hdf5(total_wo_test_ids, 'total_wo_test_ids').tolist()

        Bot_slin1_centf100 = root +'/ids/Bot_slin1_centf100.hdf5'
        self.Bot_slin1_centf100 =  self.read_hdf5(Bot_slin1_centf100, 'Bot_slin1_centf100').tolist()

        Bot_slin1_centf120 = root +'/ids/Bot_slin1_centf120.hdf5'
        self.Bot_slin1_centf120 =  self.read_hdf5(Bot_slin1_centf120, 'Bot_slin1_centf120').tolist()

        Bot_slin3_centf100 = root +'/ids/Bot_slin3_centf100.hdf5'
        self.Bot_slin3_centf100 =  self.read_hdf5(Bot_slin3_centf100, 'Bot_slin3_centf100').tolist()

        Bot_slin3_centf120 = root +'/ids/Bot_slin3_centf120.hdf5'
        self.Bot_slin3_centf120 =  self.read_hdf5(Bot_slin3_centf120, 'Bot_slin3_centf120').tolist()

        Top_slin1_centf50 = root +'/ids/Top_slin1_centf50.hdf5'
        self.Top_slin1_centf50 =  self.read_hdf5(Top_slin1_centf50, 'Top_slin1_centf50').tolist()

        Top_slin1_centf100 = root +'/ids/Top_slin1_centf100.hdf5'
        self.Top_slin1_centf100 =  self.read_hdf5(Top_slin1_centf100, 'Top_slin1_centf100').tolist()

        Top_slin1_centf120 = root +'/ids/Top_slin1_centf120.hdf5'
        self.Top_slin1_centf120 =  self.read_hdf5(Top_slin1_centf120, 'Top_slin1_centf120').tolist()

        Top_slin1_centf180 = root +'/ids/Top_slin1_centf180.hdf5'
        self.Top_slin1_centf180 =  self.read_hdf5(Top_slin1_centf180, 'Top_slin1_centf180').tolist()

        Top_slin3_centf50 = root +'/ids/Top_slin3_centf50.hdf5'
        self.Top_slin3_centf50 =  self.read_hdf5(Top_slin3_centf50, 'Top_slin3_centf50').tolist()

        Top_slin3_centf100 = root +'/ids/Top_slin3_centf100.hdf5'
        self.Top_slin3_centf100 =  self.read_hdf5(Top_slin3_centf100, 'Top_slin3_centf100').tolist()

        Top_slin3_centf120 = root +'/ids/Top_slin3_centf120.hdf5'
        self.Top_slin3_centf120 =  self.read_hdf5(Top_slin3_centf120, 'Top_slin3_centf120').tolist()

        Top_slin3_centf180 = root +'/ids/Top_slin3_centf180.hdf5'
        self.Top_slin3_centf180 =  self.read_hdf5(Top_slin3_centf180, 'Top_slin3_centf180').tolist()

        Top_slin1_centf50 = root +'/dataset_ids_v2/Top_slin1_centf50_v2_v2.hdf5'
        self.Top_slin1_centf50 =  self.read_hdf5(Top_slin1_centf50, 'Top_slin1_centf50_v2_v2').tolist()

        Top_slin1_centf100 = root +'/dataset_ids_v2/Top_slin1_centf100_v2_v2.hdf5'
        self.Top_slin1_centf100 =  self.read_hdf5(Top_slin1_centf100, 'Top_slin1_centf100_v2_v2').tolist()

        Top_slin1_centf120 = root +'/dataset_ids_v2/Top_slin1_centf120_v2_v2.hdf5'
        self.Top_slin1_centf120 =  self.read_hdf5(Top_slin1_centf120, 'Top_slin1_centf120_v2_v2').tolist()

        Top_slin1_centf180 = root +'/dataset_ids_v2/Top_slin1_centf180_v2_v2.hdf5'
        self.Top_slin1_centf180 =  self.read_hdf5(Top_slin1_centf180, 'Top_slin1_centf180_v2_v2').tolist()

        Top_slin3_centf50 = root +'/dataset_ids_v2/Top_slin3_centf50_v2_v2.hdf5'
        self.Top_slin3_centf50 =  self.read_hdf5(Top_slin3_centf50, 'Top_slin3_centf50_v2_v2').tolist()

        Top_slin3_centf100 = root +'/dataset_ids_v2/Top_slin3_centf100_v2_v2.hdf5'
        self.Top_slin3_centf100 =  self.read_hdf5(Top_slin3_centf100, 'Top_slin3_centf100_v2_v2').tolist()

        Top_slin3_centf120 = root +'/dataset_ids_v2/Top_slin3_centf120_v2_v2.hdf5'
        self.Top_slin3_centf120 =  self.read_hdf5(Top_slin3_centf120, 'Top_slin3_centf120_v2_v2').tolist()

        Top_slin3_centf180 = root +'/dataset_ids_v2/Top_slin3_centf180_v2_v2.hdf5'
        self.Top_slin3_centf180 =  self.read_hdf5(Top_slin3_centf180, 'Top_slin3_centf180_v2_v2').tolist()

        rela_pass_wd_ids = root +'/ids/rela_pass_wd_ids.hdf5'
        self.rela_pass_wd_ids =  self.read_hdf5(rela_pass_wd_ids, 'rela_pass_wd_ids').tolist()

        rela_sd_wd_ids = root +'/ids/rela_sd_wd_ids.hdf5'
        self.rela_sd_wd_ids =  self.read_hdf5(rela_sd_wd_ids, 'rela_sd_wd_ids').tolist()

        rela_stop_wd_ids = root +'/ids/rela_stop_wd_ids.hdf5'
        self.rela_stop_wd_ids =  self.read_hdf5(rela_stop_wd_ids, 'rela_stop_wd_ids').tolist()

        # self.extreme_ids = self.rela_pass_wd_ids[-3:]+self.rela_sd_wd_ids[-3:]+self.rela_stop_wd_ids[-3:]
        # self.extreme_centf_ids = self.Bot_slin1_centf100[-1:]+self.Bot_slin1_centf120[-1:]+self.Bot_slin3_centf100[-1:]+self.Bot_slin3_centf120[-1:]+self.Top_slin1_centf50[-1:]+self.Top_slin1_centf100[-1:]+self.Top_slin1_centf120[-1:]+self.Top_slin1_centf180[-1:]+self.Top_slin3_centf50[-1:]+self.Top_slin3_centf100[-1:]+self.Top_slin3_centf120[-1:]+self.Top_slin3_centf180[-1:]
        self.extreme_centf_ids2 = self.Top_slin1_centf50[-1:]+self.Top_slin1_centf100[-1:]+self.Top_slin1_centf120[-1:]+self.Top_slin1_centf180[-1:]+self.Top_slin3_centf50[-1:]+self.Top_slin3_centf100[-1:]+self.Top_slin3_centf120[-1:]+self.Top_slin3_centf180[-1:]

        to_clned_ids = root +'/ids/to_clned_ids.hdf5'
        self.to_clned_ids =  self.read_hdf5(to_clned_ids, 'to_clned_ids').tolist()

        total_ids = root +'/ids/total_ids.hdf5'
        self.total_ids =  self.read_hdf5(total_ids, 'total_ids').tolist()
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
        return len(self.extreme_centf_ids2)

    def __getitem__(self, index: int):
        # current_subfolders = self.total_wo_test_ids[index].decode('utf-8')
        # current_subfolders = self.subfolders[index]
        # current_subfolders = self.hdf5files[index]
        # current_subfolders = self.select[index]
        # current_subfolders = self.extreme_centf_ids[index].decode('utf-8')
        current_subfolders = self.extreme_centf_ids2[index].decode('utf-8')
        # current_subfolders = self.paper2[index]
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
        s_normalize = (self.s-self.mins)/(self.maxs-self.mins)-1
        # mag of s
        s_formag = self.s/20
        s_mag = 10**s_formag
        
        # add freq ids
        f_ids = torch.arange(0.2,2.1,0.1)
        s_normalize = s_normalize.unsqueeze(-1)
        s_mag = s_mag.unsqueeze(-1)
        f_ids = f_ids.unsqueeze(-1)
        s_normalize = torch.cat((s_normalize,f_ids),dim=-1)
        s_mag = torch.cat((s_mag,f_ids),dim=-1)

        # ramdomly choose 10 frequencies from 19
        f_range = range(0,19)
        f_list = list(f_range)
        f_sample = RandomSampler(f_list)
        f_random = [i for i in f_sample][0:self.num_frames]
        f_random.sort()
        f_random = torch.tensor(f_random)

        # return corresponding frames of data according to f_random
        scaledlog_cur_surfJ = scaledlog_cur_surfJ[f_random,:,:,:]
        scaledlogmag_currents = scaledlogmag_currents[f_random,:,:,:]
        s_normalize = s_normalize[f_random,:]
        s_mag = s_mag[f_random,:]

        # data enhancement by symetrical properties
        if ~(self.total_wo_test_ids[index] in self.to_clned_ids):
            if torch.rand(1)[0]>= 0.5:
                scaledlogmag_currents = np.roll(scaledlogmag_currents,int(scaledlogmag_currents.shape[2]/2),2)
                scaledlogmag_currents = np.roll(scaledlogmag_currents,int(scaledlogmag_currents.shape[3]/2),3)
                scaledlog_cur_surfJ = np.roll(scaledlog_cur_surfJ,int(scaledlog_cur_surfJ.shape[2]/2),2)
                scaledlog_cur_surfJ = np.roll(scaledlog_cur_surfJ,int(scaledlog_cur_surfJ.shape[3]/2),3)

        scaledlog_cur_surfJ = torch.tensor(scaledlog_cur_surfJ)
        scaledlogmag_currents = torch.tensor(scaledlogmag_currents)
        return scaledlog_cur_surfJ.float(), scaledlogmag_currents.float(), s_normalize.float(), s_mag.float(), scaledtopo.float(), f_random, current_subfolders

class SurfJDoubleDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.sample_folder = root + '/data/'
        self.subfolders = [f.name for f in os.scandir(self.sample_folder) if f.is_dir()]
        # load file ids
        tra_ids = root +'/dataset_ids_double/tra_ids.hdf5'
        self.tl_s =  self.read_hdf5(tra_ids, 'tl_s').tolist()
        self.tl_ol_ids =  self.read_hdf5(tra_ids, 'tl_ol_ids').tolist()

        slin1_centf100_Top = root +'/dataset_ids_double/slin1_centf100_Top.hdf5'
        self.slin1_centf100_Top_tl_s =  self.read_hdf5(slin1_centf100_Top, 'tl_s').tolist()
        self.slin1_centf100_Top_tl_ol_ids =  self.read_hdf5(slin1_centf100_Top, 'tl_ol_ids').tolist()

        slin3_centf50_Bot = root +'/dataset_ids_double/slin3_centf50_Bot.hdf5'
        self.slin3_centf50_Bot_tl_s =  self.read_hdf5(slin3_centf50_Bot, 'tl_s').tolist()
        self.slin3_centf50_Bot_tl_ol_ids =  self.read_hdf5(slin3_centf50_Bot, 'tl_ol_ids').tolist()

        slin3_centf100_Top = root +'/dataset_ids_double/slin3_centf100_Top.hdf5'
        self.slin3_centf100_Top_tl_s =  self.read_hdf5(slin3_centf100_Top, 'tl_s').tolist()
        self.slin3_centf100_Top_tl_ol_ids =  self.read_hdf5(slin3_centf100_Top, 'tl_ol_ids').tolist()

        slin3_centf180_Bot = root +'/dataset_ids_double/slin3_centf180_Bot.hdf5'
        self.slin3_centf180_Bot_tl_s =  self.read_hdf5(slin3_centf180_Bot, 'tl_s').tolist()
        self.slin3_centf180_Bot_tl_ol_ids =  self.read_hdf5(slin3_centf180_Bot, 'tl_ol_ids').tolist()


        self.extreme_centf_ids =self.slin1_centf100_Top_tl_ol_ids[-1:]+self.slin3_centf50_Bot_tl_ol_ids[-1:]+self.slin3_centf100_Top_tl_ol_ids[-1:]+self.slin3_centf180_Bot_tl_ol_ids[-1:]
        self.extreme_centf_s = self.slin1_centf100_Top_tl_s[-1:]+self.slin3_centf50_Bot_tl_s[-1:]+self.slin3_centf100_Top_tl_s[-1:]+self.slin3_centf180_Bot_tl_s[-1:]
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
        return len(self.extreme_centf_ids)

    def __getitem__(self, index: int):
        # current_subfolders = self.total_wo_test_ids[index].decode('utf-8')
        # current_subfolders = self.subfolders[index]
        current_subfolders = self.extreme_centf_ids[index]
        current_tl_folders = current_subfolders[0].decode('utf-8')
        current_ol_folders_first = current_subfolders[1].decode('utf-8')
        current_ol_folders_second = current_subfolders[2].decode('utf-8')
        current_folders = [current_tl_folders,current_ol_folders_first,current_ol_folders_second]
        # current_s = self.tl_s[index]
        current_s = self.extreme_centf_s[index]
        current_path_ol_first = self.sample_folder + current_ol_folders_first
        current_path_ol_second = self.sample_folder + current_ol_folders_second
        current_surfJ_ol_first = current_path_ol_first + '/surfj.hdf5'
        current_surfJ_ol_second = current_path_ol_second + '/surfj.hdf5'
        self.surfJ_ol_first =  self.read_hdf5(current_surfJ_ol_first, 'surfj')
        self.surfJ_ol_second =  self.read_hdf5(current_surfJ_ol_second, 'surfj')

        cur_surfJ_ol_first = np.squeeze(self.surfJ_ol_first)
        cur_surfJ_ol_first = torch.tensor(cur_surfJ_ol_first)
        cur_surfJ_ol_second = np.squeeze(self.surfJ_ol_second)
        cur_surfJ_ol_second = torch.tensor(cur_surfJ_ol_second)
        # batch_img = Image.fromarray(cur_surfJ).convert('L')
        # pipeline = torch.tensor(arr2)
        # calc log magnitudes of x-oriented and y-oriented currents
        scale = torch.log(self.maxj)
        x_r_currents_ol_first = cur_surfJ_ol_first[0:1,:,:,:]
        x_i_currents_ol_first = cur_surfJ_ol_first[1:2,:,:,:]
        y_r_currents_ol_first = cur_surfJ_ol_first[2:3,:,:,:]
        y_i_currents_ol_first = cur_surfJ_ol_first[3:4,:,:,:]
        x_mag_currents_ol_first = torch.sqrt((torch.pow(x_r_currents_ol_first,2)+torch.pow(x_i_currents_ol_first,2)))
        y_mag_currents_ol_first = torch.sqrt((torch.pow(y_r_currents_ol_first,2)+torch.pow(y_i_currents_ol_first,2)))
        x_logmag_currents_ol_first = torch.log(x_mag_currents_ol_first+1)
        y_logmag_currents_ol_first = torch.log(y_mag_currents_ol_first+1)
        logmag_currents_ol_first = torch.cat((x_logmag_currents_ol_first,y_logmag_currents_ol_first),dim=0)
        x_r_currents_ol_second = cur_surfJ_ol_second[0:1,:,:,:]
        x_i_currents_ol_second = cur_surfJ_ol_second[1:2,:,:,:]
        y_r_currents_ol_second = cur_surfJ_ol_second[2:3,:,:,:]
        y_i_currents_ol_second = cur_surfJ_ol_second[3:4,:,:,:]
        x_mag_currents_ol_second = torch.sqrt((torch.pow(x_r_currents_ol_second,2)+torch.pow(x_i_currents_ol_second,2)))
        y_mag_currents_ol_second = torch.sqrt((torch.pow(y_r_currents_ol_second,2)+torch.pow(y_i_currents_ol_second,2)))
        x_logmag_currents_ol_second = torch.log(x_mag_currents_ol_second+1)
        y_logmag_currents_ol_second = torch.log(y_mag_currents_ol_second+1)
        logmag_currents_ol_second = torch.cat((x_logmag_currents_ol_second,y_logmag_currents_ol_second),dim=0)
        # rescale
        scaledlogmag_currents_ol_first = logmag_currents_ol_first/scale
        scaledlogmag_currents_ol_second = logmag_currents_ol_second/scale
        scaledlogmag_currents_ol_first =  scaledlogmag_currents_ol_first.permute(1,0,2,3)
        scaledlogmag_currents_ol_second =  scaledlogmag_currents_ol_second.permute(1,0,2,3)

        # load topo
        current_topo_ol_first = current_path_ol_first + '/topo.hdf5'
        self.topo_ol_first =  self.read_hdf5(current_topo_ol_first, 'topo')
        cur_topo_ol_first = np.squeeze(self.topo_ol_first)
        cur_topo_ol_first = torch.tensor(cur_topo_ol_first)
        cur_topo_ol_first = cur_topo_ol_first.unsqueeze(0).unsqueeze(0)
        b = scaledlogmag_currents_ol_first.shape[0]
        d = scaledlogmag_currents_ol_first.shape[1]
        curMag_topo_ol_first = cur_topo_ol_first.repeat(b,d,1,1)
        scaledlogmag_currents_ol_first[:,:,:,:][curMag_topo_ol_first[:,:,:,:] <= 100.] = 0.
        temptopo_ol_first = torch.tensor(self.topo_ol_first)
        scaledtopo_ol_first = torch.ones(temptopo_ol_first.shape)
        scaledtopo_ol_first[temptopo_ol_first <= 100.] = 0.

        current_topo_ol_second = current_path_ol_second + '/topo.hdf5'
        self.topo_ol_second =  self.read_hdf5(current_topo_ol_second, 'topo')
        cur_topo_ol_second = np.squeeze(self.topo_ol_second)
        cur_topo_ol_second = torch.tensor(cur_topo_ol_second)
        cur_topo_ol_second = cur_topo_ol_second.unsqueeze(0).unsqueeze(0)
        b = scaledlogmag_currents_ol_second.shape[0]
        d = scaledlogmag_currents_ol_second.shape[1]
        curMag_topo_ol_second = cur_topo_ol_second.repeat(b,d,1,1)
        scaledlogmag_currents_ol_second[:,:,:,:][curMag_topo_ol_second[:,:,:,:] <= 100.] = 0.
        temptopo_ol_second = torch.tensor(self.topo_ol_second)
        scaledtopo_ol_second = torch.ones(temptopo_ol_second.shape)
        scaledtopo_ol_second[temptopo_ol_second <= 100.] = 0.

        # load s parameters
        current_s = current_s[0:181:10]
        current_s = torch.tensor(current_s)
        # mag of s
        s_formag = current_s/20
        s_mag = 10**s_formag
        
        # add freq ids
        f_ids = torch.arange(0.2,2.1,0.1)
        s_mag = s_mag.unsqueeze(-1)
        f_ids = f_ids.unsqueeze(-1)
        s_mag = torch.cat((s_mag,f_ids),dim=-1)

        # ramdomly choose 10 frequencies from 19
        f_range = range(0,19)
        f_list = list(f_range)
        f_sample = RandomSampler(f_list)
        f_random = [i for i in f_sample][0:19]
        f_random.sort()
        f_random = torch.tensor(f_random)

        # # return corresponding frames of data according to f_random
        # scaledlogmag_currents = scaledlogmag_currents[f_random,:,:,:]
        # s_normalize = s_normalize[f_random,:]

        # # data enhancement by symetrical properties
        # if ~(self.total_wo_test_ids[index] in self.to_clned_ids):
        #     if torch.rand(1)[0]>= 0.5:
        #         scaledlogmag_currents = np.roll(scaledlogmag_currents,int(scaledlogmag_currents.shape[2]/2),2)
        #         scaledlogmag_currents = np.roll(scaledlogmag_currents,int(scaledlogmag_currents.shape[3]/2),3)
        #         scaledlog_cur_surfJ = np.roll(scaledlog_cur_surfJ,int(scaledlog_cur_surfJ.shape[2]/2),2)
        #         scaledlog_cur_surfJ = np.roll(scaledlog_cur_surfJ,int(scaledlog_cur_surfJ.shape[3]/2),3)


        return scaledlogmag_currents_ol_first.float(), scaledlogmag_currents_ol_second.float(), s_mag.float(), scaledtopo_ol_first.float(), scaledtopo_ol_second.float(), f_random, current_folders


class TestSDataset(Dataset):

    def __init__(self, root, num_frames) -> None:
        super().__init__()
        self.root = root
        self.sample_folder = root
        self.test_ids = [f.name for f in os.scandir(self.sample_folder)]
        self.num_frames = num_frames

    def read_hdf5(self, file, data_name):
        with h5py.File(file, "r") as f:
            content = f[data_name][:]
        return content

    def __len__(self) -> int:
        return len(self.test_ids)

    def __getitem__(self, index: int):
        # current_subfolders = self.total_wo_test_ids[index].decode('utf-8')
        # current_subfolders = self.subfolders[index]
        current_subfolders = self.test_ids[index]
        current_path = self.sample_folder + '/'+ current_subfolders
        current_TestS = current_path
        self.TestS =  self.read_hdf5(current_TestS, 's')
        self.TestS = torch.tensor(self.TestS)
        
        # s_formag = self.TestS/20
        # s_mag = 10**s_formag
        # s_mag = s_mag[0:181:10]
        s_mag = self.TestS[0:181:10]
        
        # add freq ids
        f_ids = torch.arange(0.2,2.1,0.1)
        s_mag = s_mag.unsqueeze(-1)
        f_ids = f_ids.unsqueeze(-1)
        s_mag = torch.cat((s_mag,f_ids),dim=-1)

        # # ramdomly choose num_frames frequencies from num_freqs
        # f_range = range(0,19)
        # f_list = list(f_range)
        # f_sample = RandomSampler(f_list)
        # f_random = [i for i in f_sample][0:self.num_frames]
        # f_random.sort()
        # f_random = torch.tensor(f_random)

        # choose first num_frames frequencies from num_freqs
        f_range = range(0,19)
        f_list = list(f_range)
        f_random = [i for i in f_list][0:self.num_frames]
        f_random.sort()
        f_random = torch.tensor(f_random)

        current_subfolders = current_subfolders[:-5]

        # return corresponding frames of data according to f_random
        s_mag = s_mag[f_random,:]

        return s_mag.float(), f_random, current_subfolders

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
