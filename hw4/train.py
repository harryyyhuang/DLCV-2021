import os
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import MiniTrainDataset, PrototypicalBatchSampler
from model import ConvNet, Protoypical_loss, euclidean_dist,  sine_dist

import matplotlib.pyplot as plt

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)


# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader, dist_function=None):

    val_acc = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            # print(target)

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            # print(label_encoder)
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data
            support_feature = model(support_input.to(device))
            query_feature = model(query_input.to(device))

            # TODO: calculate the prototype for each class according to its support data
            if(dist_function != None):
                dists = dist_function(query_feature, support_feature)
            else:
                if(args.dist_function == "Euclidean"):
                    dists = euclidean_dist(query_feature, support_feature)
                elif(args.dist_function == "Sine"):
                    dists = sine_dist(query_feature, support_feature)

            # TODO: classify the query data depending on the its distense with each prototype
            out = dists.argmax(dim=-1)
            acc = (out == query_label).float().mean()

            val_acc.append(acc.item())

    avg_val_acc = np.mean(val_acc)
    print('Avg Val Acc: {}'.format(avg_val_acc))

    return avg_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=10, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--train_csv', type=str, help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, help="Training images directory")
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--dist_function', type=str, help="distance function ")

    return parser.parse_args()

def plot_loss(train_loss, val_loss, epoch, backbone):
    num = range(epoch)
    
    plt.plot(num,train_loss, label='training accuracy')
    plt.plot(num, val_loss, label='validation accuracy')
    plt.legend()
    plt.title(f'Accuracy (backbone={backbone})')
    plt.savefig(f'Accuracy{backbone}.png')


if __name__=='__main__':
    args = parse_args()

    # get device
    device = get_device()
    print(f'DEVICE: {device}')

    # test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
    #     num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
    #     sampler=GeneratorSampler(args.testcase_csv))

    train_dataset = MiniTrainDataset(args.train_csv, args.train_data_dir)

    train_loader = DataLoader(
        train_dataset, batch_sampler=PrototypicalBatchSampler(train_dataset.data_df,
                                                                      args.N_way,
                                                                      args.N_query+args.N_shot,
                                                                      iteration=600,),
                                                                      num_workers=0, 
                                                                      pin_memory=False, 
                                                                      worker_init_fn=worker_init_fn)


    test_dataset = MiniTrainDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        train_dataset, batch_sampler=PrototypicalBatchSampler(test_dataset.data_df,
                                                                      args.N_way,
                                                                      args.N_query+args.N_shot,
                                                                      iteration=600,),
                                                                      num_workers=0, 
                                                                      pin_memory=False, 
                                                                      worker_init_fn=worker_init_fn)

    model = ConvNet().to(device)
    if(args.dist_function == "Parametric"):
        proto_loss = Protoypical_loss(args.N_shot, args.N_query, device, args.N_way, dist_type="Parametric")
        optimizer = torch.optim.Adam(params=[
                {'params': model.parameters()},
                {'params': proto_loss.ds_function.parameters()}
            ],lr=0.001)
    else:
        proto_loss = Protoypical_loss(args.N_shot, args.N_query, device, args.N_way, dist_type=args.dist_function)
        optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.001)

        


    old_val_acc = 0
    train_acc_all = []
    val_acc_all = []
    for epoch in range(args.epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        print("=== Training Epoch: {} ===".format(epoch))
        train_iter = iter(train_loader)
        model.train()
        for batch in tqdm(train_iter):
            # break
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = proto_loss(model_output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss)
        avg_acc = np.mean(train_acc)
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))

        model.eval()
        print("=== Validating Epoch: {} ===".format(epoch))
        # if(args.dist_function == "Parametric"):
        #     val_acc = predict(args, model, test_loader, proto_loss.ds_function)
        # else:
        #     val_acc = predict(args, model, test_loader)
        val_iter = iter(train_loader)
        for batch in tqdm(val_iter):
            # break
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = proto_loss(model_output, y)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_val_loss = np.mean(val_loss)
        avg_val_acc = np.mean(val_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}'.format(avg_val_loss, avg_val_acc))

        train_acc_all.append(avg_acc)
        val_acc_all.append(avg_val_acc)
        

        if(avg_val_loss > old_val_acc):
            print("saveing the model for prob 1...")
            # torch.save(model.state_dict(), "feature_final_.pth")
            if(args.dist_function == "Parametric"):
                torch.save(proto_loss.ds_function.state_dict(), "ds_function.pth")
            old_val_acc = avg_val_loss

    plot_loss(train_acc_all, val_acc_all, args.epochs, "10_shot")







