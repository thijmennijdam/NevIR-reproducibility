import os
import gdown

def download_excluIR(test_manual_final_file: str, corpus_file: str):
    """
    Download the ExcluIR dataset if not already present.

    Args:
        test_manual_final_file (str): Path to the ExcluIR test manual final JSON file.
        corpus_file (str): Path to the ExcluIR corpus JSON file.
    """
    if not os.path.exists(test_manual_final_file) or not os.path.exists(corpus_file):
        if not os.path.exists(os.path.dirname(test_manual_final_file)):
            os.makedirs(os.path.dirname(test_manual_final_file))
        if not os.path.exists(os.path.dirname(corpus_file)):
            os.makedirs(os.path.dirname(corpus_file))
            
    test_manual_url ='https://drive.google.com/uc?id=1PwDeIPdGu4T2uCdhvzdVepdqdV6CzzgL'
    corpus_url = 'https://drive.google.com/uc?id=18-ODtPKGH3KC3_KijoobPHbxzDHWeUYv'
    gdown.download(test_manual_url, test_manual_final_file, quiet=False)
    gdown.download(corpus_url, corpus_file, quiet=False)
    print(f'succesfully downloaded excluIR')

if __name__ == '__main__':
    download_excluIR('data/excluIR/test_manual_final.json', 'data/excluIR/corpus.json')

