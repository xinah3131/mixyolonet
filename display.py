import os
import argparse
import copy
import csv
import os
import warnings
import cv2

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

import nets.nn as nn
import utils as util
from dataset import Dataset
# Function to save images to files
def save_images(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        image_path = os.path.join(folder_path, f'image_{i}.png')
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # Assuming image is a PyTorch tensor
        if image.dtype == numpy.float32:
            image = (image * 255).astype(numpy.uint8)
        if image.shape[2] == 1:  # If it's a grayscale image, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # If it's RGBA, remove the alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# Iterate over the loader and save the images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_path',default='yolov8_s.pt',type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    filenames = []
    gt_filenames = []
    with open('train.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(filename)
    with open('train_gt.txt') as gt_reader:
        for filename in gt_reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            gt_filenames.append(filename)

    valid_filenames = []
    valid_gt_filenames = []
    with open('test.txt') as valid_reader:
        for filename in valid_reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            valid_filenames.append(filename)
    with open('test_gt.txt') as valid_gt_reader:
        for filename in valid_gt_reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            valid_gt_filenames.append(filename)

    dataset = Dataset(filenames, gt_filenames, args.input_size, params, True)
    valid_dataset = Dataset(valid_filenames,valid_gt_filenames,args.input_size, params, True)
    if args.world_size <= 1:
        sampler = None
        valid_sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)
        valid_sampler = data.distributed.DistributedSampler(valid_dataset)
    dataset = Dataset(filenames,gt_filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, args.batch_size, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    for batch_idx, batch in enumerate(loader):
        images, gt_images, _, _ = batch
        save_images(gt_images, f'batch_{batch_idx}_images')
        break  # Save only the first batch of images

if __name__ == "__main__":
    main()
