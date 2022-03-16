
from numpy.core.numeric import zeros_like
from data_loader_voxel import LunaVoxelTestDataset

import torch
from torch.utils.data import DataLoader
from torch.nn import Unfold

from pytorch3dunet.unet3d.model import ResidualUNet3D, ResidualUNetFeatureExtract

from train import parse_args

from tqdm import tqdm

import random
import numpy as np

import csv


from scipy.ndimage import measurements, morphology
from util import irc2xyz

# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def get_voxel_data(candidate, ct_pre=None):
    # initialize tensor to return
    # voxel_z_size = 32
    # voxel_x_size = 48
    # voxel_y_size = 48
    voxel_size = [32, 48, 48]
    # if(ct_pre == None):
    #     ct = CT_test(candidate.series_uid)
    # else:
    ct = ct_pre
    hu_t = ct.hu_a
    # center_irc = xyz2irc(
    #         candidate.center_xyz,
    #         ct.origin_xyz,
    #         ct.vxSize_xyz,
    #         ct.direction_a,
    #     )
    center_irc = candidate

    slice_list = []
    for axis, center_val in enumerate(center_irc):
        start_ndx = int(round(center_val - voxel_size[axis]/2))
        end_ndx = int(start_ndx + voxel_size[axis])

        # assert center_val >= 0 and center_val < hu_a.shape[axis], repr([series_uid, center_xyz, origin_xyz, vxSize_xyz, center_irc, axis])

        if start_ndx < 0:
            # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
            #     series_uid, center_xyz, center_irc, hu_a.shape, voxel_size))
            start_ndx = 0
            end_ndx = int(voxel_size[axis])
        
        if end_ndx > ct.hu_a.shape[axis]:
            # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
            #     series_uid, center_xyz, center_irc, hu_a.shape, voxel_size))
            end_ndx = ct.hu_a.shape[axis]
            start_ndx = int(ct.hu_a.shape[axis] - voxel_size[axis])

        slice_list.append(slice(start_ndx, end_ndx))

    hu_t = hu_t[tuple(slice_list)].clip(-1000, 1000)
    return torch.tensor(hu_t)


def groupSegmentationOutput(series_uid,  ct, clean_a, pred_a, model_cl, object_device):
        
        print("start segmentationing")
        # # erosion operation
        clean_a = morphology.binary_erosion(clean_a, iterations=1)
        # # dilation operation
        # clean_a = morphology.binary_dilation(clean_a, iterations=5)

        # connected-components algorithm to group the pixels
        candidateLabel_a, candidate_count = measurements.label(clean_a)

        # 求 probability
        prob_list = []
        labels = np.unique(candidateLabel_a)
        # for i in labels[1:]:
        #     mask = np.where(candidateLabel_a == i)
        #     # print(np.unique(output_a[mask]))
        #     prob_false = np.mean(pred_a[0][mask].detach().cpu().numpy())
        #     prob = np.mean(pred_a[1][mask].detach().cpu().numpy())
        #     prob_list.append(round(prob, 2))
            
        
        objects = measurements.find_objects(candidateLabel_a)
        # print(clean_a[8:121, 228:408, 68:474])
        # print(objects)
        # print(ct.vxSize_xyz)
        # exit()


        # 求質心代表結節位置 
        centerIrc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001, # 輸入不能為複數所以+1001
            labels=candidateLabel_a,
            index=np.arange(1, candidate_count+1),
        )

        # print(len(centerIrc_list))
        candidateInfo_list = []
        prob_filt_list = []
        for i, center_irc in tqdm(enumerate(centerIrc_list), total=len(centerIrc_list)): # IRC to XYZ

            per_voxel = get_voxel_data(center_irc, ct)
            per_out = model_cl(per_voxel.unsqueeze(0).unsqueeze(0).to(object_device))
            if(per_out[0][1] < 0.9):
                continue
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            # object_irc = objects[i]
            # i_length = object_irc[0].stop - object_irc[0].start
            # r_length = object_irc[1].stop - object_irc[1].start
            # c_length = object_irc[2].stop - object_irc[2].start
            # volumn = i_length*ct.vxSize_xyz.z + r_length*ct.vxSize_xyz.x + c_length*ct.vxSize_xyz.y
            # if(volumn < 50):
            #     continue
            # exit()
            # candidateInfo_tup = CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz)
            print(series_uid, center_xyz, per_out[0][1])
            candidateInfo_list.append((series_uid, center_xyz))
            prob_filt_list.append(float(per_out[0][1].detach().cpu().numpy()))
        # exit()
        return (candidateInfo_list, prob_filt_list)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

if __name__=='__main__':
    device = get_device()
    print(f'DEVICE: {device}')

    args = parse_args()

    test_dataset = LunaVoxelTestDataset()
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = ResidualUNet3D(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load("3D-Unet.pth"))
    model_classifier = ResidualUNetFeatureExtract(in_channels=1, out_channels=2).to(device)
    model_classifier.load_state_dict(torch.load("3D-Unet-classifier.pth"))
    model.eval()
    model_classifier.eval()
    predict_all = []

    for data in tqdm(test_dataset):
        data_voxel = data[0].squeeze()

        data_voxel_fold = data_voxel.unfold(0, 32, 32).unfold(1, 96, 96).unfold(2, 96, 96)
        min_num = data[0][0][0][0]
        data_voxel_batch = data_voxel_fold.reshape(-1, 1, data_voxel_fold.shape[3], data_voxel_fold.shape[4], data_voxel_fold.shape[5])

        data_voxel_out = torch.zeros((2, data_voxel_batch.shape[0]*data_voxel_batch.shape[1],  data_voxel_fold.shape[3], data_voxel_fold.shape[4], data_voxel_fold.shape[5]), dtype=torch.float32)

        # print(f"process {i}")
        for i, batch_data in enumerate(data_voxel_batch):
            # print(batch_data.shape)
            if(torch.sum(batch_data != min_num) == 0):
                # print(batch_data)   
                continue
            # print(batch_data != min_num)
            # print(torch.sum(batch_data == 0.8194))
            # print(f"not passing {i}")
            segmentation = model(batch_data.unsqueeze(1).to(device))
            if hasattr(model, 'final_activation') and model.final_activation is not None:
                        segmentation = model.final_activation(segmentation)
            data_voxel_out[:, i*1:(i+1)*1, :, :, :] = segmentation.detach().clone().transpose(0, 1)

        data_voxel_out = data_voxel_out.reshape(-1, data_voxel_fold.shape[0], data_voxel_fold.shape[1], data_voxel_fold.shape[2], data_voxel_fold.shape[3], data_voxel_fold.shape[4], data_voxel_fold.shape[5])
        inference_voxel = torch.ones((2, data_voxel.shape[0], data_voxel.shape[1], data_voxel.shape[2]))

        for i in range(data_voxel_fold.shape[0]):
            for r in range(data_voxel_fold.shape[1]):
                for c in range(data_voxel_fold.shape[2]):


                    inference_voxel[:, i*32:(i+1)*32, r*96:(r+1)*96, c*96:(c+1)*96] = data_voxel_out[:, i, r, c]

        pred = groupSegmentationOutput(data[1], data[2], inference_voxel.argmax(0), inference_voxel, model_classifier, device)
        predict_all.append(pred)

        # for p in pred:
        #     print(p)
        # print(len(pred))
        # break
        # print(unfold(data[0]).shape)
        # ret = 
        # exit()
    with open('out_3d_Unet.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        field = ['seriesuid','coordX','coordY','coordZ','probability']
        csv_writer.writerow(field)
        csvfile.flush()
        for pred, prob in predict_all:
             for i, candidate in enumerate(pred):
                data = []
                # print(candidate)
                center_xyz = (candidate[1].x, candidate[1].y, candidate[1].z)
                proba = prob[i]
                series_uid = candidate[0]
                # s = f"center xyz {center_xyz}, probability {prob}"
                # print(s)

                data.append(series_uid)
                data.append(center_xyz[0])
                data.append(center_xyz[1])
                data.append(center_xyz[2])
                data.append(proba)
                csv_writer.writerow(data)
                csvfile.flush()
                

