import os
import torch

from tqdm import tqdm
from config.config import Config
from database.db_data import DatabaseData
from database.source_data import get_source_loader
from model.cath_model import CATHPredictorModel


class DatabaseBuilder(object):
    def __init__(self):
        self.config = Config()
        self.model = CATHPredictorModel(load_base_model=False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.loader = get_source_loader()
        self.load_model()
        self.database_data = DatabaseData()

    def load_model(self):
        model_path = self.config.cath_model_path
        assert os.path.exists(model_path), f"Model not found: {model_path}"
        self.model.load_state_dict(torch.load(model_path), strict=False)

    def build_database(self):
        self.model.eval()

        for batch in tqdm(self.loader, total=len(self.loader)):
            batch.to(self.device)
            embeddings, saml = self.model.get_embeddings(batch)
            embeddings_ = embeddings.cpu().detach().numpy()
            saml_ = torch.argmax(saml, dim=-1)
            saml_ = saml_.cpu().detach().numpy()
            indices_ = batch.data_idx.cpu().detach().numpy()
            batch_ids = batch.batch.cpu().detach().numpy()
            self.database_data.append_batch(embeddings_,
                                            saml_,
                                            indices_,
                                            batch_ids)

        self.database_data.save_data()


def build_database():
    dbBuilder = DatabaseBuilder()
    dbBuilder.build_database()


if __name__ == '__main__':
    build_database()
