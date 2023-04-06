import os
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select
from config.config import Config


class ChainSelect(Select):
    dx = 8

    def __init__(self, chains, positions=None):
        self.chains = chains
        self.current_chain = None
        self.positions = positions
        self.residues_idx = 0

    def accept_model(self, model):
        if model.id == 0:
            return 1

    def accept_chain(self, chain):
        if chain.get_id() in self.chains:
            self.current_chain = chain.get_id()
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        if self.positions is None:
            return True
        self.residues_idx += 1

        if self.positions[0] <= self.residues_idx <= self.positions[1]:
            return True
        return False


def extract_(filepath, chain_id, positions, cath):
    config = Config()
    output_folder = config.test_folder
    assert os.path.exists(output_folder)
    output_folder = output_folder / Path(filepath).stem
    output_folder.mkdir(exist_ok=True)
    output_folder = output_folder / cath
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / f'{Path(filepath).stem}_{chain_id}{positions[0]}_{chain_id}{positions[1]}.pdb'

    parser = PDBParser(PERMISSIVE=1)
    assert os.path.exists(filepath)
    structure = parser.get_structure(id=0, file=str(filepath))
    io_w_no_h = PDBIO()
    io_w_no_h.set_structure(structure)
    io_w_no_h.save(str(output_file), ChainSelect(chains=[chain_id], positions=positions))
    return output_file


def extract_results(filepath, chain, results):
    outputs = []

    for items in results:
        for record in items:
            cath_, positions = record
            outfile = extract_(filepath, chain, positions, cath_)
            outputs.append(outfile)
    return outputs
