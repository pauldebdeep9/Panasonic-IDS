import os
import pickle
from pathlib import Path
import pandas as pd


def predictions_to_graph(prediction_file, graph_folder):
    
    file = open(prediction_file,'rb')
    f_infos = pickle.load(file)
    file.close()

    Path(graph_folder).mkdir(parents=True, exist_ok=True)

    headers = ['cause','effect','source','support','evidence']

    rows = []
    for (cause,effect), evidences in f_infos.items():
        for evidence in evidences:
            if ('unicausal' in evidence['method']) or ('causenet' in evidence['method']): # add filtering condition
                rows.append([
                    cause,
                    effect,
                    "f-"+str(evidence['doc_id']),#+"-"+str(evidence['sent_id']),
                    len(evidence['method']), #support
                    evidence['cause_span']+' --> '+evidence['effect_span']+'; '+str(evidence['method'])+'; '+str(evidence['text'])
                ])
            
    graph_df = pd.DataFrame(rows, columns=headers)
    # get nodes
    node_df = pd.concat([
        graph_df[['cause','source']].rename(columns={'cause':'node','source':'sources'}),
        graph_df[['effect','source']].rename(columns={'effect':'node','source':'sources'})
    ],axis=0).drop_duplicates().reset_index(drop=True)
    node_df = node_df.groupby(['node'])['sources'].apply(lambda x: ', '.join(x)).reset_index()

    # raw graph (before clustering)
    graph_df.to_csv(os.path.join(graph_folder,'graph.csv'), index=False, encoding='utf-8-sig')
    node_df.to_csv(os.path.join(graph_folder,'node_df.csv'), index=False, encoding='utf-8-sig')
    print(f"complete - predictions_to_graph")


if __name__ == "__main__":
    prediction_file = r"D:\66 CausalMap\Panasonic-IDS\outs\20230313_MIR_rels.obj"    
    graph_folder = r"D:\66 CausalMap\SciLit_CausalMap\visualization\mir_paper2"
    predictions_to_graph(prediction_file,graph_folder)
    # STOPPED HERE, NEED TO RERUN TO GET NUMBERS EXCLUDING + & M