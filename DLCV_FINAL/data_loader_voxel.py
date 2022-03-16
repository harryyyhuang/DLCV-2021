# ------------------------------------------------------------------------
# LunaDataset
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from dlwpt-code (https://github.com/deep-learning-with-pytorch/dlwpt-code)
# Copyright (c). All Rights Reserved
# ------------------------------------------------------------------------
import csv
import functools
import glob
import os 
import copy
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np
from data_loader import Luna2dDataset, getCt

import torch
from torch.utils.data import Dataset

from util import getCache
from util import XyzTuple, IrcTuple
from util import xyz2irc, irc2xyz

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import calculate_stats

import matplotlib.pyplot as plt

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import ball, disk, binary_erosion, binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi
from tqdm import tqdm

# raw_cache = getCache('train_raw')

DataInfoTuple = namedtuple(
    'DataInfoTuple',
    'diameter_mm, series_uid, center_xyz',
)


CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)
def getDataList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob("data/test/*.mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # diameter_dict = {}
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

        mhd_path = glob.glob(
            'data/test/{}.mhd'.format(series_uid)
        )[0]


        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        # ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        dataInfo_list = getDataInfoDict()[self.series_uid]

        # mask for possible true label
        self.positive_mask = self.buildAnnotationMask(dataInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2))
                                 .nonzero()[0].tolist())

    # algorithm to extract mask.
    # see more in dlwpt-code
    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)
        # boundingBox_a = np.zeros((2, self.hu_a.shape[0], self.hu_a.shape[1], self.hu_a.shape[2]), dtype=np.bool)
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


    def getCenterData(self, center_xyz):
        return xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )[0]

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





##### for classification training and validation dataset #####
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob("data/test/*.mhd")
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


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

# @raw_cache.memoize(typed=True)
def getCtRawData(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunck, pos_chunk, center_irc = ct.getRawData(center_xyz, width_irc)
    ct_chunck.clip(-1000, 1000, ct_chunck)
    return ct_chunck, pos_chunk, center_irc


class LunaVoxelDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 data_static=None):

        # Get all the data name in train dir 
        self.series_list = sorted(getDataInfoDict().keys())

        # Get all data info
        self.dataInfo_list = getDataList()

        # # Get data static
       # if(data_static == None):
       #     self.data_static = self.getStatic()
       # else:
       #     self.data_static = data_static
        # self.data_static = (-1000.0, 1000.0, -512.2683, 486.98715)

        # f = open("static.txt", 'w')
        # print(str(self.data_static[0]), file=f)
        # print(str(self.data_static[1]), file=f)
        # print(str(self.data_static[2]), file=f)
        # print(str(self.data_static[3]), file=f)
        # f.close()
        # self.data_static = (-3024.0, 17739.0, -796.2107, 971.68866)
        # Record if validation
        self.isValSet_bool = isValSet_bool





        # Get val if set validation delete data vice versa
        if(isValSet_bool):
            self.dataInfo_list = self.dataInfo_list[::val_stride]
            config = {
            'raw': [{'name': 'ToTensor', 'expand_dims': True}],
            'label': [ {'name': 'ToTensor', 'expand_dims': True, 'dtype': 'long'}],}
        else:
            del self.dataInfo_list[::val_stride]
            config = {
            'raw': [{'name' : 'RandomFlip'}, {'name' : 'AdditiveGaussianNoise'}, {'name' : 'AdditivePoissonNoise'},
                    {'name': 'ElasticDeformation', 'spline_order': 3}, {'name': 'ToTensor', 'expand_dims': True}],
            'label': [{'name' : 'RandomFlip'}, {'name': 'ElasticDeformation', 'spline_order': 0}, 
                     {'name': 'ToTensor', 'expand_dims': True, 'dtype': 'long'}],
            }

        transformer = transforms.get_transformer(config, 0, 0, 0, 0)
        self.raw_transformer = transformer.raw_transform()
        self.mask_transformer = transformer.label_transform()


        

        print("{!r}: {} {} series, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.dataInfo_list),
        ))

    def __len__(self):
        return len(self.dataInfo_list)

    # we return data of size 7x64x64
    def __getitem__(self, index):
        if(self.isValSet_bool):
            data = self.getitem_fullCT(self.dataInfo_list[index])
        else: 
            data = self.getitem_cropCT(self.dataInfo_list[index])

        mask = self.mask_transformer(data[1])
        mask_one_hot = torch.zeros(2, mask.shape[1], mask.shape[2], mask.shape[3])
        mask_one_hot.scatter_(0, mask, 1)


        return self.raw_transformer(data[0]), mask_one_hot, data[2]

    def getStatic(self):
        print("============ Calculate data static ===============")
        ct_all = []
        for data_info in self.dataInfo_list:
            ct_all.append(getCt(data_info.series_uid).hu_a.clip(-1000, 1000))

        return calculate_stats(ct_all)


    def getitem_fullCT(self, dataInfo_tup):
        # ct = getCt(dataInfo_tup.series_uid)
        # ct_t = torch.zeros((32, 256, 256))
        # pos_t = torch.zeros((32, 256, 256))

        # slice_ndx =  ct.getCenterData(dataInfo_tup.center_xyz)

        # start_ndx = slice_ndx - 16
        # end_ndx = slice_ndx + 16
        # for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
        #     context_ndx = max(context_ndx, 0)
        #     context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
        #     ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))[256-128:256+128, 256-128:256+128]
        #     pos_t[i] = torch.from_numpy(ct.positive_mask[context_ndx].astype(np.float32))[256-128:256+128, 256-128:256+128]

        # ct_t.clamp_(-1000, 1000)

        # return ct_t.unsqueeze(0), pos_t.unsqueeze(0), dataInfo_tup.series_uid
        ct_a, pos_a, center_irc = getCtRawData(
            dataInfo_tup.series_uid,
            dataInfo_tup.center_xyz,
            (32, 96, 96),
        )
        ct_t = ct_a.astype(np.float32)
        pos_t = pos_a.astype(np.long)
        return ct_t, pos_t, dataInfo_tup.series_uid


    def getitem_cropCT(self, dataInfo_tup):
        ct_a, pos_a, center_irc = getCtRawData(
            dataInfo_tup.series_uid,
            dataInfo_tup.center_xyz,
            (32, 96, 96),
        )

        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        ct_t = ct_a[:, row_offset:row_offset+64,
                                     col_offset:col_offset+64].astype(np.float32)
        pos_t = pos_a[:, row_offset:row_offset+64,
                                       col_offset:col_offset+64].astype(np.long)



        return ct_t, pos_t, dataInfo_tup.series_uid



class CT_test:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data/test/{}.mhd'.format(series_uid)
        )[0]

        
        ct_mhd = sitk.ReadImage(mhd_path)

        
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.hu_a = self.hu_a.clip(-1000, 1000, self.hu_a)
        

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        # ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)


    def segment_lung(self, im, plot=True):
        if plot == True:
            f, plots = plt.subplots(8, 1, figsize=(5, 40))

        binary = im < -400
        if plot == True:
            plots[0].axis('off')
            plots[0].imshow(binary, cmap=plt.cm.bone) 

        cleared = clear_border(binary)
        
        if plot == True:
            plots[1].axis('off')
            plots[1].imshow(cleared, cmap=plt.cm.bone) 


        label_image = label(cleared)
        if plot == True:
            plots[2].axis('off')
            plots[2].imshow(label_image, cmap=plt.cm.bone) 

        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        label_image[coordinates[0], coordinates[1]] = 0

        binary = label_image > 0
        if plot == True:
            plots[3].axis('off')
            plots[3].imshow(binary, cmap=plt.cm.bone) 

        selem = disk(2)
        binary = binary_erosion(binary, selem)
        if plot == True:
            plots[4].axis('off')
            plots[4].imshow(binary, cmap=plt.cm.bone) 

        selem = disk(10)
        binary = binary_closing(binary, selem)
        if plot == True:
            plots[5].axis('off')
            plots[5].imshow(binary, cmap=plt.cm.bone) 

        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)
        if plot == True:
            plots[6].axis('off')
            plots[6].imshow(binary, cmap=plt.cm.bone) 

        get_high_vals = binary == 0
        im[get_high_vals] = -1000
        if plot == True:
            plots[7].axis('off')
            plots[7].imshow(im, cmap=plt.cm.bone) 
        if plot == True:
            f.savefig('preprocess.png')
        return im

    def remove_largest(self, hu_seg):
        selem = ball(2)
        binary = binary_closing(hu_seg, selem)

        label_scan = label(binary)
        areas = [r.area for r in regionprops(label_scan)]
        areas.sort()

        for r in regionprops(label_scan):
            max_x, max_y, max_z = 0, 0, 0
            min_x, min_y, min_z = hu_seg.shape[0], hu_seg.shape[1], hu_seg.shape[2]
            
            for c in r.coords:
                max_z = max(c[0], max_z)
                max_y = max(c[1], max_y)
                max_x = max(c[2], max_x)
                
                min_z = min(c[0], min_z)
                min_y = min(c[1], min_y)
                min_x = min(c[2], min_x)
            if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
                for c in r.coords:
                    hu_seg[c[0], c[1], c[2]] = 0

        return hu_seg
        

class LunaVoxelTestDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 data_static=None):

        # Get all the data name in train dir 
        self.series_list = self.get_series_list()
        

        # # Get data static
        # if(data_static == None):
        #     self.data_static = self.getStatic()
        # else:
        #     self.data_static = data_static
        # self.data_static = (-3024.0, 17739.0, -796.2107, 971.68866)

        config = {
            'raw': [{'name': 'ToTensor', 'expand_dims': False}],}


        transformer = transforms.get_transformer(config, 0, 0, 0,0)
        self.raw_transformer = transformer.raw_transform()


        

        print("{!r}: {} series".format(
            self,
            len(self.series_list),
        ))

    @functools.lru_cache(1, typed=True)
    def get_series_list(self):
        mhd_list = glob.glob("data/test/*.mhd")
        # series_list = []
        # with open('/home/yihan/yihan/Final/data/sample_seriesuids.csv', newline='') as f:
        #     reader = csv.reader(f)
        #     for data in reader:
        #         series_list.append(data[0])

        # print(series_list)
        series_list = [os.path.split(p)[-1][:-4] for p in mhd_list]
        return series_list

    def __len__(self):
        return len(self.series_list)

    # we return data of size 7x64x64
    def __getitem__(self, index):
        data = self.getitem_fullCT(self.series_list[index])

        return self.raw_transformer(data[0]), data[1], data[2]
        # return data[0], data[1], data[2]


    def getStatic(self):
        print("============ Calculate data static ===============")
        ct_all = []
        for data_info in self.dataInfo_list:
            ct_all.append(getCt(data_info.series_uid).hu_a)

        return calculate_stats(ct_all)


    def getitem_fullCT(self, series_uid):
        ct = self.getCt(series_uid)
        ct_t = ct.hu_a.astype(np.float32)
        # ct_t.clip(-1000, 1000, ct_t)
        return ct_t, series_uid, ct

    @functools.lru_cache(1, typed=True)
    def getCt(self, series_uid):
        return CT_test(series_uid)


    

class LunaFalsePositiveDataset(Dataset):
    def __init__(self,
                val_stride=0,
                isValSet_bool=None,
                sortby_str='series_uid',
                ratio_int=0,
                aug_dict=None,
                per_batch_size = 8,
                voxel_size = (32, 48, 48)
        ):
        
        self.ratio_int = ratio_int
        self.aug_dict = aug_dict
        self.candidateInfo_list = getCandidateInfoList()
        self.per_batch_size = per_batch_size
        self.voxel_size = voxel_size
        self.isValSet_bool = isValSet_bool
        if(isValSet_bool):
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            config = {'raw':[{'name': 'ToTensor', 'expand_dims': False}],
            # 'label': [ {'name': 'ToTensor', 'expand_dims': True, 'dtype': 'long'}],
            }
        else:
            del self.candidateInfo_list[::val_stride]
            config = {'raw':[{'name' : 'RandomFlip'}, {'name' : 'AdditiveGaussianNoise'}, {'name' : 'AdditivePoissonNoise'},
                    {'name': 'ElasticDeformation', 'spline_order': 3}, {'name': 'ToTensor', 'expand_dims': False}],
            # 'label': [{'name' : 'RandomFlip'}, {'name': 'ElasticDeformation', 'spline_order': 0}, 
            #          {'name': 'ToTensor', 'expand_dims': True, 'dtype': 'long'}],
            }
        
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
        # self.neg_voxels = self.preprocess_pos_voxel(self.negative_list)
        self.negative_series_index = self.get_neg_index(self.negative_list)
        
        # augmentation
        # config = {'raw' : [{'name' : 'RandomFlip'},
        #             {'name': 'ElasticDeformation', 'spline_order': 3}, {'name': 'ToTensor', 'expand_dims': False}]}
        train_transforms = transforms.get_transformer(config, 0, 0, 0, 0)
        self.transforms = train_transforms.raw_transform()

    def preprocess_pos_voxel(self, candidate_list):
        voxel_z_size = self.voxel_size[0]
        voxel_x_size = self.voxel_size[1]
        voxel_y_size = self.voxel_size[2]
        all_pos_voxels = torch.zeros((len(candidate_list), voxel_z_size, voxel_x_size, voxel_y_size), dtype=torch.float32)
        for i, candidate_tup in tqdm(enumerate(candidate_list)):
            all_pos_voxels[i] = self.get_voxel_data(candidate_tup)

        return all_pos_voxels
   

    def __len__(self):
        if(self.isValSet_bool):
            return len(self.negative_series_index)
        else:
            return 2000

    def __getitem__(self, index):
        index = index%len(self.negative_series_index)
        voxel_z_size = self.voxel_size[0]
        voxel_x_size = self.voxel_size[1]
        voxel_y_size = self.voxel_size[2]

        per_data = torch.zeros((self.per_batch_size, voxel_z_size, voxel_x_size, voxel_y_size), dtype=torch.float32)
        per_data_label = torch.zeros((self.per_batch_size), dtype=torch.long)

        # positive sample
        pos_sample_index = torch.randperm(self.pos_voxels.shape[0])[:self.per_batch_size//2]
        pos_sample_voxel = self.pos_voxels[pos_sample_index]
        per_data[:self.per_batch_size//2] = pos_sample_voxel
        per_data_label[:self.per_batch_size//2] = 1

        # negative sample
        # neg_sample_index = torch.randperm(self.neg_voxels.shape[0])[:self.per_batch_size//2]
        # neg_sample_voxel = self.neg_voxels[neg_sample_index]
        # per_data[self.per_batch_size//2:] = neg_sample_voxel
        # per_data_label[self.per_batch_size//2:] = 0

        # sample negative sample
        start_idx = self.negative_series_index[index]
        find_end = lambda index : self.negative_series_index[index+1] if index+1 < len(self.negative_series_index) else len(self.negative_list)
        end_idx = find_end(index)
        # choices = slice(np.random.choice(len(self.negative_list[start_idx: end_idx]), self.per_batch_size//2, replace=True).astype(int))
        # print(self.negative_list[start_idx: end_idx])
        neg_sample_list = random.sample(self.negative_list[start_idx: end_idx], self.per_batch_size//2)
        # print(choices)
        # neg_sample_list = self.negative_list[start_idx: end_idx][choices]
        # neg_sample_list = self.negative_list[start_idx: start_idx+2]
        ct = CT_test(neg_sample_list[0].series_uid)
        for i, neg_candidate_tup in enumerate(neg_sample_list):
            per_data[i+self.per_batch_size//2] = self.get_voxel_data(neg_candidate_tup, ct_pre=ct)
            per_data_label[i+self.per_batch_size//2] = 0

        permute_index =  torch.randperm(per_data.shape[0])
        per_data = per_data[permute_index].clamp_(-1000, 1000)
        per_data_label = per_data_label[permute_index]
        per_data = self.transforms(per_data.detach().cpu().numpy())

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
            ct = CT_test(candidate.series_uid)
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

        hu_t = hu_t[tuple(slice_list)].clip(-1000, 1000)
        return torch.tensor(hu_t)



if __name__ == "__main__":
    data = LunaFalsePositiveDataset(val_stride=10, isValSet_bool=True)
    for one in data:
        a_1, a_2 = one
        print(a_1.shape)
    #     print(post.shape)





        

        
