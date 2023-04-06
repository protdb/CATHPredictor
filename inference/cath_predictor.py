import os
import pickle
import numpy as np
import torch
import faiss

from config.config import Config
from inference.pdb_loader import batch_from_file
from model.cath_model import CATHPredictorModel
from utils.utils import convert_to_sa
from Bio import pairwise2


class CathPredictor(object):
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = CATHPredictorModel(load_base_model=False)
        self.model.to(self.device)
        self.load_model()
        self.model.eval()
        self.fs_index = None
        self.data_indices = None
        self.sa_data = {}
        self.sa_metadata = {}
        self.load_metadata()

    def load_model(self):
        model_path = self.config.cath_model_path
        assert os.path.exists(model_path), f"Model not found: {model_path}"
        self.model.load_state_dict(torch.load(model_path), strict=False)

    def load_metadata(self):
        assert os.path.exists(self.config.data_file), f"File not found: {self.config.data_file}"
        assert os.path.exists(self.config.index_file), f"File not found: {self.config.index_file}"
        assert os.path.exists(self.config.sa_data_file), f"File not found: {self.config.sa_data_file}"

        self.fs_index = faiss.read_index(str(self.config.data_file))

        with open(self.config.index_file, 'rb') as fh:
            self.data_indices = pickle.load(fh)

        with open(self.config.sa_data_file, 'rb') as fh:
            self.sa_data = pickle.load(fh)

        self.__reorder_metadata()

    def predict_cath(self, filepath, chain):
        embedding, sa = self.__get_file_embeddings(filepath, chain)
        candidates = self.search_cath(embedding)
        results = self.__inference(candidates, sa)
        return results

    def __reorder_metadata(self):

        for idx in self.data_indices:
            index_record = self.data_indices[idx]
            v = list(index_record.values())[0]
            if len(v) == 1:
                sa = self.sa_data[idx]
                cath_ = v[0]
                if cath_ in self.sa_metadata:
                    self.sa_metadata[cath_].append(sa)
                else:
                    self.sa_metadata.update({cath_: [sa]})

    @torch.no_grad()
    def __get_file_embeddings(self, filepath, chain):
        assert os.path.exists(filepath), f"File not found {filepath}"
        data = batch_from_file(filepath, chain)
        data.to(self.device)
        embedding, saml = self.model.get_embeddings(data)
        saml_ = torch.argmax(saml, dim=-1)
        sa_idx = saml_.cpu().detach().numpy()
        sa = convert_to_sa(sa_idx)
        embedding_ = embedding.detach().cpu().numpy()

        return embedding_, sa

    def search_cath(self, query, k=5):
        distances, indices = self.fs_index.search(query, k)
        candidates = []
        prefer_candidates = []

        for i in range(k):
            data_idx = indices[0][i]
            distance = distances[0][i]
            candidates.append(data_idx)

            if distance <= self.config.distance_patience:
                prefer_candidates.append(data_idx)

        if prefer_candidates:
            candidates = prefer_candidates

        return candidates

    def __inference(self, candidates, sa):
        scores = []
        cath_ids = []

        for data_idx in candidates:
            candidates_sa = self.sa_data[data_idx]
            cath_ = self.data_indices[data_idx]
            score = self.__sa_score(sa, candidates_sa)
            scores.append(score)
            cath_ids.append((data_idx, cath_))

        if len(scores) > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        scored = {}
        u_idx = []
        for i, score in enumerate(scores):
            if score > self.config.score_patience:
                cath_ = list(cath_ids[i][1].values())[0]
                if cath_ not in u_idx:
                    scored.update({score: (cath_ids[i][0], cath_)})
                u_idx.append(cath_)

        scored_keys = list(scored.keys())
        scored_keys.sort(reverse=True)
        scored = [(scored[k]) for k in scored_keys]

        results = []

        for item in scored:
            r_ = self.__inference_region(item, sa)
            if r_:
                results.append(r_)

        return results

    @staticmethod
    def __sa_score(source_sa, target_sa):
        align = pairwise2.align.globalms(source_sa, target_sa, 2, -1, -1, -1, score_only=True)
        return align / len(target_sa)

    def __inference_region(self, item, source_sa):
        data_idx, cath_ = item
        if len(cath_) == 1:
            target_sa = self.sa_data[data_idx]
            best_score, posS, posE = self.__sa_region(source_sa, target_sa)
            if (posE - posS) <= self.config.cut_factor:
                return []

            results = [(cath_[0], (posS, posE))]

        else:
            results = []

            for cl in cath_:
                if cl not in self.sa_metadata:
                    continue
                saml = self.sa_metadata[cl]
                best_score = -1
                posS = -1
                posE = -1
                sax = saml if len(saml) < self.config.align_samples else \
                    np.random.choice(saml, self.config.align_samples)

                for sa in sax:
                    score, pos1, pos2 = self.__sa_region(source_sa, sa)

                    if score > best_score:
                        best_score = score
                        posS = pos1
                        posE = pos2

                if best_score > 0 and (posE - posS) >= self.config.cut_factor:
                    results.append((cl, (posS, posE)))
        return results

    def __sa_region(self, source_sa, target_sa):
        align = pairwise2.align.localms(source_sa, target_sa, 2, -1, -1, -1)
        best_score = -1
        start_position = -1
        end_position = - 1
        for a in align:
            if a.score > best_score:
                best_score = a.score
                start_position, end_position = a.start, a.end

        assert start_position != -1 and end_position != -1
        dx = self.config.cut_sa_dx

        start_position = start_position - dx if start_position > dx else 0
        end_position = end_position + dx if start_position + end_position - dx < len(source_sa) \
            else start_position + len(source_sa) + dx // 2

        return best_score / len(target_sa), start_position, end_position


