import os
import pandas as pd
from tqdm import tqdm
from log_reader import LogReader
from overlap_tools import retain_intersection
from report_tools import report_counts, report_intersection
import pickle
from pathlib import Path


def main(csv_file_path, folder):
    # Process
    print('Processing...')
    f_infos = {}
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for log_file_path in tqdm(file_paths):
        s,e = log_file_path.split("MIR-")[-1].split(".log")[0].split("_")
        reader = LogReader(log_file_path, csv_file_path, causenet_simplify=True)
        retain_intersection(reader, f_infos, s)
    
    # Reporting
    report_counts(f_infos)
    report_intersection(f_infos)

    # Remove duplicates
    print('De-duplicating...')
    for keypair in f_infos.keys():
        if len(f_infos[keypair])>1:
            tmp = pd.DataFrame(f_infos[keypair])
            tmp['tmp'] = tmp['text'].apply(lambda x: x.replace(" ", "").lower())
            f_infos[keypair] = [f_infos[keypair][i] for i in tmp.drop_duplicates(subset='tmp').index]

    # Reporting
    report_counts(f_infos)
    report_intersection(f_infos)

    # Store
    file_name = Path(csv_file_path).stem
    filehandler = open(os.path.join(r"D:\66 CausalMap\Panasonic-IDS\outs",f"{file_name}_rels.obj"),"wb")
    pickle.dump(f_infos,filehandler)
    filehandler.close()


if __name__ == "__main__":
    """
    conda activate cre
    python process_extraction
    """
    csv_file_path = r"D:\66 CausalMap\Panasonic-IDS\data\MIR.csv"
    folder = r"D:\79 Mass\outs\MIR"
    main(csv_file_path, folder)


