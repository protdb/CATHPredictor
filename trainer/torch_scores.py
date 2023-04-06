import sys

import numpy as np
from torchmetrics.classification import MultilabelAccuracy, MultilabelRecall, MultilabelPrecision, MultilabelF1Score
from data_loader.cath_dataset import N_LEVEL_CLASSES, N_CLASSES


class Scores(object):
    def __init__(self, output_file=None):
        self.output_file = output_file
        self.triplet_losses = []

    def push_bath_metrics(self, triplet_loss):
        self.triplet_losses.append(triplet_loss.cpu().detach().item())

    def out_metrics(self, phase, epoch):
        self.print_message(f'Phase: {phase}: Epoch: {epoch} '
                           f'Triplet loss: {np.mean(self.triplet_losses):2f}')
        self.print_message('')

    def print_message(self, msg):
        original_stdout = sys.stdout
        if self.output_file is not None:
            with open(str(self.output_file), 'a') as f:
                sys.stdout = f
                print(msg)
        else:
            print(msg)
        sys.stdout = original_stdout
