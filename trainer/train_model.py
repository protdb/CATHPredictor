import os
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from config.config import Config
from data_loader.cath_aggegate import get_loaders
from model.cath_model import CATHPredictorModel
import torch.nn as nn

from trainer.torch_scores import Scores


class TrainerBase(object):
    def __init__(self):
        self.config = Config()
        self.batch_size = self.config.batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.loader = get_loaders()
        self.model = CATHPredictorModel().float()
        self.model.to(self.device)
        self.n_epochs = self.config.train_epochs
        self.model_path = self.config.cath_model_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.MSELoss()
        self.scores = None

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path), strict=False)
            print(f'Model loaded {self.model_path}')
        else:
            print(f'Model not found {self.model_path}')

    def save_model(self):
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.model_path)

    def one_step(self, item, metric_obj=None):
        loss = self.model(item)

        if metric_obj is not None:
            metric_obj.push_bath_metrics(loss)

        return loss

    def train_model(self):
        best_result = 1e6
        self.model.train()

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            metrics = Scores(output_file=self.config.metric_file)

            for sample in tqdm(self.loader, total=len(self.loader)):
                sample = [item.to(self.device) for item in sample]
                self.optimizer.zero_grad()
                loss = self.one_step(sample, metrics)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
                self.optimizer.step()
                total_loss += loss.cpu().detach().item()

            metrics.out_metrics(phase='train', epoch=epoch)

            if total_loss < best_result:
                best_result = total_loss
                self.save_model()


def train_model():
    trainer = TrainerBase()
    trainer.load_model()
    trainer.train_model()


if __name__ == '__main__':
    train_model()
