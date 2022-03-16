import csv
import functools
import glob
import os 
import copy
import math 
from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from util import getCache
from util import XyzTuple, IrcTuple
from util import xyz2irc, irc2xyz

import random

from pytorch3dunet.augment import transforms


raw_cache = getCache('train_raw')

DataInfoTuple = namedtuple(
    'DataInfoTuple',
    'diameter_mm, series_uid, center_xyz'
)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

##### for segmentation testing dataset #####
def getTestDataList():
    mhd_list = glob.glob("data/test/*.mhd")
    test_uid_list = [os.path.split(p)[-1][:-4] for p in mhd_list]

    return test_uid_list


class TestCt:
    def __init__(self, series_uid):
        mhd_path = glob.glob('data/test/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)


    def getRawData(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):

            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


def getTestCt(series_uid):
    return TestCt(series_uid)


class LunaTestDataset(Dataset):
    def __init__(self, series_uid):
        # Get all the data name in test dir 
        # self.series_list = getTestDataList()
        # print("{!r}: {} series".format(
        #     self,
        #     len(self.series_list),
        # ))

        self.series_uid = series_uid
        self.contextSlices_count = 3

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.getitem_fullCT(self.series_uid)

    def getitem_fullCT(self, series_uid): 
        ct = self.getCt(series_uid)
        ct_t = torch.zeros((ct.hu_a.shape[0], self.contextSlices_count * 2 + 1, 512, 512))
        
        for slice_ndx in range(ct_t.shape[0]):
            start_ndx = slice_ndx - self.contextSlices_count
            end_ndx = slice_ndx + self.contextSlices_count + 1
        
            for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
                context_ndx = max(context_ndx, 0)
                context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
                ct_t[slice_ndx][i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_t.clip_(-1000, 1000)

        return ct_t, series_uid, ct, slice_ndx

    def getCt(self, series_uid):
        return TestCt(series_uid)


##### for segmentation training and validation dataset #####
@functools.lru_cache(1)
def getDataList(requireOnDisk_bool=True):
    mhd_list = glob.glob("data/train/*.mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    dataInfo_list = []
    with open("data/annotations.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            dataInfo_list.append(DataInfoTuple(
                annotationDiameter_mm,
                series_uid,
                annotationCenter_xyz,
            ))

    dataInfo_list.sort(reverse=True)
    return dataInfo_list


@functools.lru_cache(1)
def getDataInfoDict(requireOnDisk_bool=True):
    dataInfo_list = getDataList(requireOnDisk_bool)
    dataInfo_dict = {}

    for dataInfo_tup in dataInfo_list:
        dataInfo_dict.setdefault(dataInfo_tup.series_uid,
                                      []).append(dataInfo_tup)
    return dataInfo_dict


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('data/train/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        candidateInfo_list = getDataInfoDict()[self.series_uid]

        self.positive_mask = self.buildAnnotationMask(candidateInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2))
                                 .nonzero()[0].tolist())

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawData(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):

            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawData(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunck, pos_chunk, center_irc = ct.getRawData(center_xyz, width_irc)
    ct_chunck.clip(-1000, 1000, ct_chunck)
    return ct_chunck, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False):

        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getDataInfoDict().keys())


        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]


        self.dataInfo_list = getDataList()

        # print("{!r}: {} {} series, {} slices, {} nodules".format(
        #     self,
        #     len(self.series_list),
        #     {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
        #     len(self.sample_list),
        #     len(self.dataInfo_list),
        # ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dDataset(Luna2dDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.dataInfo_list)
        random.shuffle(self.sample_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.dataInfo_list[ndx % len(self.dataInfo_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        ct_a, pos_a, center_irc = getCtRawData(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        
        pos_a = pos_a[3:4]

        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64,
                                     col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64,
                                       col_offset:col_offset+64]).to(torch.long)

        slice_ndx = center_irc.index
        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx 



##### for classification training and validation dataset #####
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob("data/train/*.mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm),
            )

    candidateInfo_list = []
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0

            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                # candidateDiameter_mm = annotationDiameter_mm

                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


def getCtAugmentedCandidate(augmentation_dict,series_uid, center_xyz, width_irc):
    ct = getTestCt(series_uid)
    ct_chunk, center_irc = ct.getRawData(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float


    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaFalsePositiveDataset(Dataset):
    def __init__(self,
                val_stride=0,
                isValSet_bool=None,
                sortby_str='series_uid',
                ratio_int=0,
                aug_dict=None,
                per_batch_size = 4,
                voxel_size = (32, 48, 48)
        ):
        
        self.ratio_int = ratio_int
        self.aug_dict = aug_dict
        self.candidateInfo_list = getCandidateInfoList()
        self.per_batch_size = per_batch_size
        self.voxel_size = voxel_size


        # split validation data
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        
        # sort candidate list
        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        
        # create neg and pos list
        self.negative_list = [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.pos_voxels = self.preprocess_pos_voxel(self.pos_list)
        self.negative_series_index = self.get_neg_index(self.negative_list)
        
        # augmentation
        # config = {'raw' : [{'name' : 'RandomFlip'},
        #             {'name': 'ElasticDeformation', 'spline_order': 3}, {'name': 'ToTensor', 'expand_dims': False}]}
        # train_transforms = transforms.get_transformer(config, 0, 0, 0, 0)
        # self.transforms = train_transforms.raw_transform()

    def preprocess_pos_voxel(self, candidate_list):
        voxel_z_size = self.voxel_size[0]
        voxel_x_size = self.voxel_size[1]
        voxel_y_size = self.voxel_size[2]
        all_pos_voxels = torch.zeros((len(candidate_list), voxel_z_size, voxel_x_size, voxel_y_size), dtype=torch.float32)
        for i, candidate_tup in enumerate(candidate_list):
            all_pos_voxels[i] = self.get_voxel_data(candidate_tup)

        return all_pos_voxels
   

    def __len__(self):
        return len(self.negative_series_index)

    def __getitem__(self, index):

        voxel_z_size = self.voxel_size[0]
        voxel_x_size = self.voxel_size[1]
        voxel_y_size = self.voxel_size[2]

        per_data = torch.zeros((self.per_batch_size, voxel_z_size, voxel_x_size, voxel_y_size), dtype=torch.float32)
        per_data_label = torch.zeros((self.per_batch_size, 2), dtype=torch.long)

        # positive sample
        pos_sample_index = torch.randperm(self.pos_voxels.shape[0])[:self.per_batch_size//2]
        pos_sample_voxel = self.pos_voxels[pos_sample_index]
        per_data[:self.per_batch_size//2] = pos_sample_voxel
        per_data_label[:self.per_batch_size//2, 1] = 1

        # sample negative sample
        start_idx = self.negative_series_index[index]
        find_end = lambda index : self.negative_series_index[index+1] if index+1 < len(self.negative_series_index) else len(self.negative_list)
        end_idx = find_end(index)
        neg_sample_list = random.sample(self.negative_list[start_idx: end_idx], self.per_batch_size//2)
        # neg_sample_list = self.negative_list[start_idx: start_idx+2]
        ct = getTestCt(neg_sample_list[0].series_uid)
        for i, neg_candidate_tup in enumerate(neg_sample_list):
            per_data[i+self.per_batch_size//2] = self.get_voxel_data(neg_candidate_tup, ct_pre=ct)
            per_data_label[i+self.per_batch_size//2, 0] = 1

        permute_index =  torch.randperm(per_data.shape[0])
        per_data = per_data[permute_index].clamp_(-1000, 1000)
        per_data_label = per_data_label[permute_index]
        
        return per_data, per_data_label


    def get_neg_index(self, candidate_list):
        candidate_index = [0]
        old_candidate = candidate_list[0]
        for i, candidate in enumerate(candidate_list):
            if(candidate.series_uid != old_candidate.series_uid):
                candidate_index.append(i)
                old_candidate = candidate
        return candidate_index

    def get_voxel_data(self, candidate, ct_pre=None):
        # initialize tensor to return
        voxel_z_size = self.voxel_size[0]
        voxel_x_size = self.voxel_size[1]
        voxel_y_size = self.voxel_size[2]
        if(ct_pre == None):
            ct = getTestCt(candidate.series_uid)
        else:
            ct = ct_pre
        hu_t = ct.hu_a
        center_irc = xyz2irc(
                candidate.center_xyz,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - self.voxel_size[axis]/2))
            end_ndx = int(start_ndx + self.voxel_size[axis])

            # assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, self.voxel_size))
                start_ndx = 0
                end_ndx = int(self.voxel_size[axis])
            
            if end_ndx > ct.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, self.voxel_size))
                end_ndx = ct.hu_a.shape[axis]
                start_ndx = int(ct.hu_a.shape[axis] - self.voxel_size[axis])

            slice_list.append(slice(start_ndx, end_ndx))


        # hu_t = hu_t[max(center_irc.index-voxel_z_size//2, 0):min(center_irc.index+voxel_z_size//2, hu_t.shape[0]),
        #             max(center_irc.row-voxel_x_size//2, 0):min(center_irc.row+voxel_x_size//2, hu_t.shape[1]),
        #             max(center_irc.col-voxel_y_size//2, 0):min(center_irc.col+voxel_y_size//2, hu_t.shape[2])].clip(-1000, 1000)

        # hu_t = np.pad(hu_t, self.getPadSize(hu_t.shape), mode='constant', constant_values=(-1000, -1000))
        hu_t = hu_t[tuple(slice_list)].clip(-1000, 1000)
        return torch.tensor(hu_t)


    def getPadSize(self, hu_shape):
        pad_sizes = []
        for i in range(3):
            pad_size = self.voxel_size[i] - hu_shape[i]

            if(pad_size != 0):
                pad_sizes.append((pad_size//2, pad_size-pad_size//2))
            else:
                pad_sizes.append((0,0))

        return tuple(pad_sizes)

    def getCtAugmentedCandidate(augmentation_dict, series_uid, center_xyz, width_irc):
        ct = getCt(series_uid)
        ct_chunk, pos_chunk, center_irc = ct.getRawData(center_xyz, width_irc)

        ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

        transform_t = torch.eye(4)

        for i in range(3):
            if 'flip' in augmentation_dict:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            if 'offset' in augmentation_dict:
                offset_float = augmentation_dict['offset']
                random_float = (random.random() * 2 - 1)
                transform_t[i,3] = offset_float * random_float

            if 'scale' in augmentation_dict:
                scale_float = augmentation_dict['scale']
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if 'rotate' in augmentation_dict:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

            transform_t @= rotation_t

        affine_t = F.affine_grid(
                transform_t[:3].unsqueeze(0).to(torch.float32),
                ct_t.size(),
                align_corners=False,
            )

        augmented_chunk = F.grid_sample(
                ct_t,
                affine_t,
                padding_mode='border',
                align_corners=False,
            ).to('cpu')

        if 'noise' in augmentation_dict:
            noise_t = torch.randn_like(augmented_chunk)
            noise_t *= augmentation_dict['noise']

            augmented_chunk += noise_t

        return augmented_chunk[0], center_irc



##### for classification testing dataset #####
class LunaTestFalsePositiveDataset(Dataset):
    def __init__(self,
                 series_uid=None,
                 voxel_size = (32, 48, 48),
                 candidateInfo_list=None,
            ):

        self.candidateInfo_list = copy.copy(candidateInfo_list)
        # self.series_list = sorted(set(candidateInfo_tup.series_uid 
        #                               for candidateInfo_tup in self.candidateInfo_list))
        self.voxel_size = voxel_size
        self.ct = getTestCt(series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)
    
    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
                
        # ct = getTestCt(candidateInfo_tup.series_uid)
        candidate_a, center_irc = self.ct.getRawData(
            candidateInfo_tup.center_xyz,
            self.voxel_size,
        )
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        return candidate_t, torch.tensor(center_irc)



###### for showing CT images #####
def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.savefig("annotation_validation.png")


# if __name__ == "__main__":
    # data_test = TrainingLuna2dDataset()
    # data_test = Luna2dDataset()
    # index = 10
    # print("image shape: ", data_test[index][0].shape)
    # print("mask shape (0 or 1): ", data_test[index][1].shape)
    # print("uid: ", data_test[index][2])
    # print("slice index: ", data_test[index][3])
    
    # img = data_test[index][0]
    # mask = data_test[index][1]
    

    # mhd_path = f'data/train/{data_test[index][2]}.mhd'
    # ct_mhd = sitk.ReadImage(mhd_path)
    # idxSlice = data_test[index][3]
    # sitk_show(ct_mhd[:,:,idxSlice])