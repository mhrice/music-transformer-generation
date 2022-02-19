

import os
import csv
import shutil
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data.sampler import SubsetRandomSampler
from model import MusicTransformer
from dataset import REMI_dataset
from utils.loss import SmoothCrossEntropyLoss
from utils.constants import *
from utils.lr_scheduling import LrStepTracker, get_lr
from utils.argument_funcs import parse_train_args, print_train_args, write_model_params
from utils.run_model import train_epoch, eval_model

CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Eval loss", "Eval accuracy"]

def main():
    parser = argparse.ArgumentParser(description='Preprocess midi files for training')
    parser.add_argument("dataset", help="dataset directory")
    parser.add_argument("out", help="output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.out, "model_params.txt")
    # write_model_params(args, params_file)

    weights_folder = os.path.join(args.out, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.out, "results")
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_acc_file = os.path.join(results_folder, "best_acc_weights.pickle")
    best_text = os.path.join(results_folder, "best_epochs.txt")

    dataset = REMI_dataset(args.dataset)
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    model = MusicTransformer(n_layers=n_layers, num_heads=num_heads,
                d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                max_sequence=max_sequence, rpr=rpr)


    start_epoch = -1
    ##### Lr Scheduler vs static lr #####
    init_step = 0
    lr = LR_DEFAULT_START
    lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    train_loss_func = eval_loss_func

    ##### Optimizer #####
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    lr_scheduler = LambdaLR(opt, lr_stepper.step)

    ##### Tracking best evaluation accuracy #####
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)


    ##### TRAIN LOOP #####
    epochs = 100
    for epoch in range(start_epoch, epochs):
        print(SEPERATOR)
        print("NEW EPOCH:", epoch+1)
        print(SEPERATOR)
        print("")

        # Train
        train_epoch(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler, 1)

        print(SEPERATOR)
        print("Evaluating:")

        # Eval
        train_loss, train_acc = eval_model(model, train_loader, train_loss_func)
        eval_loss, eval_acc = eval_model(model, validation_loader, eval_loss_func)

        # Learn rate
        lr = get_lr(opt)

        print("Epoch:", epoch+1)
        print("Avg train loss:", train_loss)
        print("Avg train acc:", train_acc)
        print("Avg eval loss:", eval_loss)
        print("Avg eval acc:", eval_acc)
        print(SEPERATOR)
        print("")

        new_best = False

        if(eval_acc > best_eval_acc):
            best_eval_acc = eval_acc
            best_eval_acc_epoch  = epoch+1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if(eval_loss < best_eval_loss):
            best_eval_loss       = eval_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                print("Best eval acc epoch:", best_eval_acc_epoch, file=o_stream)
                print("Best eval acc:", best_eval_acc, file=o_stream)
                print("")
                print("Best eval loss epoch:", best_eval_loss_epoch, file=o_stream)
                print("Best eval loss:", best_eval_loss, file=o_stream)


        if((epoch+1) % 1 == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch+1, lr, train_loss, train_acc, eval_loss, eval_acc])

if __name__ == "__main__":
    main()

## remi
## output_dir = "checkpoints"
