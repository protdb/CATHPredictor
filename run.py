from extract.extractor import extract_results
from inference.cath_predictor import CathPredictor


def test_predictor(filepath, chain):

    predictor = CathPredictor() # Loads the model and metadata. Better call once
    results = predictor.predict_cath(filepath, chain)

    for items in results:
        for record in items:
            cath_, positions = record
            print(f"CATH: {cath_} AA Positions: {positions}")

    # For test: Extracts folds from the source file and puts them in config.test_folder
    extract_results(filepath=file,
                    chain=chain,
                    results=results)


input_pdb_id = '1a2f'

if __name__ == '__main__':
    file = f'/home/dp/Data/PDB/{input_pdb_id.lower()}.pdb'

    test_predictor(file, 'A')
