import os
import torch
import sys

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS

def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)

def manage_checkpoints(args, colbert, optimizer, batch_idx, epoch):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')
    print(f'in manage checkpoint with batch_ix = {batch_idx}')

    if not os.path.exists(path):
        os.mkdir(path)

    # sien edited this code: original =
    # if batch_idx % 2000 == 0:
    if batch_idx % 58 == 0:
        print(f'save checkpoint {batch_idx}')
        #name = os.path.join(path, "colbert.dnn")
        # name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        name = os.path.join(path, "colbert-{}.dnn".format(epoch))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)