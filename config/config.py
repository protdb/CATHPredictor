import os.path
from pathlib import Path


class Config(object):
    def __init__(self):
        self.preprocessed_folder = "/home/dp/Data/SAML/preprocessed_pdb"
        self.working_folder = "/home/dp/Data/cath_predictor"
        self.log_folder = "log"
        self.metric_file = "cath_metrics.log"
        self.model_folder = "models"

        self.database_folder = "database"
        self.test_folder = "test"
        self.index_file = 'index.pkl'
        self.data_file = 'data.fs'
        self.sa_data_file = 'sa.pkl'

        self.batch_size = 2
        self.num_workers = 8
        self.train_epochs = 24
        self.distance_patience = 1.5
        self.score_patience = 0.7
        self.cut_factor = 16
        self.align_samples = 20
        self.cut_sa_dx = 5
        self.base_model_path = "abbnet"
        self.cath_model_path = "cath.pth"
        self.create_workspace()

    def create_workspace(self):
        self.working_folder = Path(self.working_folder)
        assert os.path.exists(self.working_folder)

        self.log_folder = self.working_folder / self.log_folder
        self.log_folder.mkdir(exist_ok=True)
        self.metric_file = self.log_folder / self.metric_file

        self.model_folder = self.working_folder / self.model_folder
        self.model_folder.mkdir(exist_ok=True)
        self.base_model_path = self.model_folder / self.base_model_path
        self.cath_model_path = self.model_folder / self.cath_model_path

        self.database_folder = self.working_folder / self.database_folder
        self.database_folder.mkdir(exist_ok=True)
        self.index_file = self.database_folder / self.index_file
        self.data_file = self.database_folder / self.data_file
        self.sa_data_file = self.database_folder / self.sa_data_file

        self.test_folder = self.working_folder / self.test_folder
        self.test_folder.mkdir(exist_ok=True)


class EncoderConfig(object):
    def __init__(self):
        self.input_dim = 256
        self.h_dim = 128
        self.n_layers = 3
        self.alpha = 0.5,
        self.beta = 1.,
        self.gamma = .1
