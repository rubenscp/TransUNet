import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse, trainer_white_mold

from wm_utils import WM_Utils

# setting the environment variables for CUDA and DSA
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='research/white-mold-applications/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='research/white-mold-applications/TransUNet/lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

print(f'train.py 00 - before main')

if __name__ == "__main__":
    print(f'train.py 01 - args: {args}')
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    if dataset_name == 'Synapse':
        dataset_config = {
            'Synapse': {
                'root_path': 'research/white-mold-applications/project_TransUNet/data/Synapse/train_npz',
                'list_dir': 'research/white-mold-applications/TransUNet/lists/lists_Synapse',
                'num_classes': 9,
            },
        }
    else:   
        if dataset_name == 'WhiteMold':
            dataset_config = {
                'WhiteMold': {
                    'root_path': '/home/lovelace/proj/proj939/rubenscp/research/white-mold-dataset/results-pre-processed-images/running-0021-15ds-300x300-merged-classes/splitting_by_images/4-balanced-output-dataset/mask-image',
                    'list_dir': '',
                    'num_classes': 8,
                },
            }

    print(f'train.py dataset_config: {dataset_config}')
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "research/white-mold-applications/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    print(f'snapshot_path: {snapshot_path}')

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]

    print(f'')  
    print(f'config_vit: {config_vit}')
    config_vit.pretrained_path = 'research/white-mold-applications/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'

    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # print(f'model: {net}')
    net.load_from(weights=np.load(config_vit.pretrained_path))
    print(f'')
    print(f'model after load_from: {net}')

    if dataset_name == 'Synapse':
        trainer = {'Synapse': trainer_synapse,}
    else:   
        if dataset_name == 'WhiteMold':
            trainer = {'WhiteMold': trainer_white_mold,}

    # creating working directory for images and masks 
    path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/TransUNet/results/train'
    WM_Utils.remove_directory(path)        
    WM_Utils.create_directory(path)

    print(f'train.py - dataset_name: {dataset_name}')
    print(f'train.py - trainer[dataset_name]: {trainer[dataset_name]}')
    print(f'train.py - args: {args}')
    # print(f'train.py - net: {net}')
    print(f'train.py - snapshot_path: {snapshot_path}')
    trainer[dataset_name](args, net, snapshot_path)
    print(f'train.py 02 - after trainer[dataset_name](args, net, snapshot_path)')
    print(f'Finished training')