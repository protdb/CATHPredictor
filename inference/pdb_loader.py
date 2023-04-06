import io
import os
import warnings
import numpy as np
from Bio.PDB import PDBParser, Select, PDBIO, PDBExceptions
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from data_loader.geometric_features import ProteinFeatures

warnings.filterwarnings("ignore", category=PDBExceptions.PDBConstructionWarning)

BACKBONE_MASK = ['N', 'CA', 'C', 'O']


class PDBBackbone(object):
    def __init__(self, pdb_path, pdb_id=0):
        self.filename = pdb_path
        self.structure = PDBParser().get_structure(pdb_id, self.filename)
        self.model = None

    def extract_features(self):
        sequence = []
        backbone_coo = []
        for res in self.structure.get_residues():
            res_name = res.get_resname()
            try:
                res_coo = []
                for el in BACKBONE_MASK:
                    coo = res[el].get_coord()
                    res_coo.append(coo)
            except KeyError:
                continue
            sequence.append(res_name)
            backbone_coo.append(res_coo)

        assert sequence
        assert backbone_coo
        backbone_coo = np.array(backbone_coo)
        return backbone_coo, sequence


class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = chains
        self.current_chain = None

    def accept_model(self, model):
        if model.id == 0:
            return 1

    def accept_chain(self, chain):
        if chain.get_id() in self.chains:
            self.current_chain = chain.get_id()
            return True
        else:
            return False

    def accept_residue(self, residue):
        return True

    def accept_atom(self, atom):
        return True if atom.get_name() in BACKBONE_MASK else False


class PDBSelector(object):
    def __init__(self):
        self.parser = PDBParser(PERMISSIVE=1)

    def extract_chains(self, path, chain_id):
        assert os.path.exists(path)
        structure = self.parser.get_structure(id=0, file=str(path))
        output = io.StringIO()
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        io_w_no_h.save(output, ChainSelect(chain_id))
        output.seek(0)
        return output


class DataFromFile(Dataset):
    def __init__(self, file, chain):
        assert os.path.exists(file)
        self.file = file
        self.chain = chain
        self.protein_features = ProteinFeatures()

    def __getitem__(self, index):
        samples = PDBSelector()
        target_path = samples.extract_chains(self.file, self.chain)
        pdb_extractor = PDBBackbone(target_path)
        try:
            features = pdb_extractor.extract_features()
            data = self.protein_features.build_features(features)
            assert features is not None
            assert features[0] is not None
            assert features[1] is not None
        except AssertionError:
            return None
        return data

    def __len__(self):
        return 1


def batch_from_file(file, chain):
    file_ds = DataFromFile(file, chain)
    loader = DataLoader(file_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1)
    batch = next(iter(loader))
    if type(batch) == list:
        return None
    return batch
