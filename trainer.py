import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets_tun.dataset_synapse import * # import all functions from dataset_synapse.py
from datasets_tun.dataset_white_mold import * # import all functions from dataset_white_mold.py

from wm_utils import WM_Utils

def trainer_synapse(args, model, snapshot_path):
    # from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    print(f'trainer synapse args.root_path: {args.root_path}')
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    print(f'trainer before dataloader Synapse')
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    print(f'trainer after dataloader Synapse - len(trainloader): {len(trainloader)}')

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            print(f'trainer_synapse image_batch.shape: {image_batch.shape}')
            print(f'trainer_synapse label_batch.shape: {label_batch.shape}')
            outputs = model(image_batch)
            
            # saving image and label for debugging
            predicted_image = outputs[0][0].cpu().detach().numpy()
            print(f'trainer_synapse predicted_image.shape: {predicted_image.shape}')
            predicted_label = label_batch[0].cpu().detach().numpy()
            print(f'trainer_synapse predicted_label.shape: {predicted_label.shape}')
            path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/TransUNet/results/train'       
            print(f'class Synapse_dataset, __getitem__, path: {path}')        
            image_filename = 'predicted_output_0_0_image.jpeg'
            path_and_filename = os.path.join(path, image_filename)
            print(f'trainer_synapse path_and_filename: {path_and_filename}')
            WM_Utils.save_image(path_and_filename, predicted_image)
            WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'predicted_image', predicted_image)
            image_filename = 'predicted_output_0_label.jpeg'
            path_and_filename = os.path.join(path, image_filename)
            print(f'trainer_synapse path_and_filename: {path_and_filename}')
            WM_Utils.save_image(path_and_filename, predicted_label)
            WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'predicted_label', predicted_label)
            # end of saving image and label for debugging

            print(f'trainer_synapse outputs.shape: {outputs.shape}')
            print(f'trainer_synapse outputs[0]: {outputs[0]}')
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished of the Synapse dataset!"

def trainer_white_mold(args, model, snapshot_path):
    # print(f'trainer_whitemold args.root_path: {args.root_path}')
    # print(f'trainer_whitemold model: {model}')
    # print(f'trainer_whitemold snapshot_path: {snapshot_path}')
    
    # from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log_white_mold.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # print(f'trainer white mold args.root_path: {args.root_path}')
    db_train = WhiteMold_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                 transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # print(f'trainer before dataloader WhiteMold')
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, 
                             pin_memory=True, worker_init_fn=worker_init_fn)
    # print(f'trainer after dataloader WhiteMold - len(trainloader): {len(trainloader)}')
    
    # print(f'')
    # print(f'trainer - properties of trainloader')
    # print(f'trainer - trainloader.dataset: {trainloader.dataset}')
    # print(f'trainer - trainloader.batch_size: {trainloader.batch_size}')
    # print(f'trainer - trainloader.num_workers: {trainloader.num_workers}')
    # print(f'trainer - trainloader.pin_memory: {trainloader.pin_memory}')
    # print(f'trainer - trainloader.batch_sampler: {trainloader.batch_sampler}')

    # print(f'')
    # print(f'trainer - args.n_gpu: {args.n_gpu}')

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    losses = []

    # print(f'trainer - len(trainloader): {len(trainloader)}')
    # item_trainloader = next(iter(trainloader))
    # print(f'trainer - item_trainloader: {item_trainloader}')
        
    for epoch_num in iterator:
        print(f'')
        print(f'trainer - epoch_num: {epoch_num}')
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            
            # # saving image and label for debugging
            # predicted_image = outputs[0][0].cpu().detach().numpy()
            # print(f'trainer_white_mold predicted_image.shape: {predicted_image.shape}')
            # predicted_label = label_batch[0].cpu().detach().numpy()
            # print(f'trainer_white_mold predicted_label.shape: {predicted_label.shape}')
            # path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/TransUNet/results/train'       
            # print(f'class WhiteMold_dataset , __getitem__, path: {path}')        
            # image_filename = 'predicted_output_0_0_image.jpeg'
            # path_and_filename = os.path.join(path, image_filename)
            # print(f'trainer_white_mold path_and_filename: {path_and_filename}')
            # WM_Utils.save_image(path_and_filename, predicted_image)
            # WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'predicted_image', predicted_image)
            # image_filename = 'predicted_output_0_label.jpeg'
            # path_and_filename = os.path.join(path, image_filename)
            # print(f'trainer_white_mold path_and_filename: {path_and_filename}')
            # WM_Utils.save_image(path_and_filename, predicted_label)
            # WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'predicted_label', predicted_label)
            # # end of saving image and label for debugging

            # print(f'após treinar um batch - 01')
            # print(f'trainer_white_mold outputs.shape: {outputs.shape}')
            # print(f'trainer_white_mold outputs: {outputs}')
            # print(f'após treinar um batch - 02')
            # print(f'trainer_white_mold label_batch.shape: {label_batch.shape}')
            # print(f'trainer_white_mold label_batch: {label_batch}')
            # print(f'trainer_white_mold label_batch[:].long(): {label_batch[:].long()}')
            # print(f'após treinar um batch - 03')

            # print(f'trainer_white_mold checking max value of the class ids')
            for idx in range(label_batch.shape[0]):
                max_val = torch.max(label_batch[idx])
                min_val = torch.min(label_batch[idx])
                if max_val > 5:
                    print(f'label_batch[{idx}] - max: {max_val}, min: {min_val}')
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('epoch %d iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # saving losses per epoch
        losses.append([epoch_num+1, iter_num, loss.item(), loss_ce.item(), loss_dice.item()])

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()

    # saving training losses
    print(f'losses: {losses}')
    path_and_filename_losses = os.path.join(snapshot_path, "losses.xlsx")
    WM_Utils.save_losses(losses, path_and_filename_losses)

    return "Training Finished of the White Mold dataset!"

