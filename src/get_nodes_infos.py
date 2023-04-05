import nltk
from nltk.tag.stanford import StanfordNERTagger
import os
import re
import pandas as pd
from tqdm import tqdm
from copy import copy

java_path = r"C:\Program Files\AdoptOpenJDK\jdk-11.0.7.10-hotspot\bin\java.exe"
PATH_TO_JAR= r"C:\Users\effbl\stanza_corenlp\stanford-ner-2020-11-17\stanford-ner.jar"
PATH_TO_MODEL = r"C:\Users\effbl\stanza_corenlp\stanford-ner-2020-11-17\classifiers\english.muc.7class.distsim.crf.ser.gz"

os.environ['JAVAHOME'] = java_path
tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL,path_to_jar=PATH_TO_JAR, encoding='utf-8')

DEMO_SIZE = None
template = {
    'ORGANIZATION': [],
    'LOCATION': [],
    'DATE': [],
    'TIME': [],
    'ACTION': [],
    'OBJECT': []
}

def get_nodes_infos(graph_folder):
    """
    Get NER information of each node span

    Args:
        graph_folder (str): Path to folder that contains "graph.csv"
    """
    # load graph infos
    csv_file_path = os.path.join(graph_folder,'graph.csv')
    data = pd.read_csv(csv_file_path)
    if DEMO_SIZE is not None:
        data = data[:int(DEMO_SIZE/2)]

    print(f'working on: {csv_file_path}')
    print(f"length of data: {len(data)}")

    batch_size = 200
    n_batches = (len(data)//batch_size)+1
    continue_from = None
    save_file_name = os.path.join(graph_folder,f"nodes_extracted_infos.pkl")

    if continue_from is None:
        final = pd.DataFrame()
        loop = range(n_batches)
    else:
        final = pd.read_pickle(save_file_name)
        print(f"reloaded 'final' from: {save_file_name}")
        loop = range(continue_from,n_batches)

    for batch_id in loop:
        print(f'batch: {batch_id} / {n_batches}')
        subset = parse(data[batch_id*batch_size:(batch_id+1)*batch_size])
        final = pd.concat([final,subset],axis=0)
        final.to_pickle(save_file_name) # backup
    
    print(f"file saved to: {save_file_name}")
    print(f"length of processed data: {len(final)}")
    print(f"complete - get_nodes_infos")


def parse(data):
    
    cause_action = []
    effect_action = []
    cause_action_rem = []
    effect_action_rem = []
    evidence_ner = []
    cause_store = []
    effect_store = []

    whitespace = '\s*'
    specials = ['*','+','(',')','[',']'] # need to escape special regex chars
    
    for i,row in tqdm(data.iterrows(), total=data.shape[0]):
        cause = row.cause
        effect = row.effect
        cause_rem = cause
        effect_rem = effect
        sentence = row.evidence    
        words = nltk.word_tokenize(sentence) 
        tagged = tagger.tag(words)
        tagged2 = []
        curr_word = ''
        curr_ner = None
        for (word,ner) in tagged:
            if ner==curr_ner: # same, continue
                curr_word+=str(word)
            else: # different
                if curr_ner is not None:
                    tagged2.append((curr_word,curr_ner))
                # reset
                if ner=='O':
                    curr_ner=None
                    curr_word=''
                else:
                    curr_ner=ner
                    curr_word=word

        cause_template = copy(template)
        effect_template = copy(template)
        for (word,ner) in tagged2:
            if ner!='O':
                word = word.lower()
                pattern = ''.join(["\\"+str(c)+whitespace if c in specials else str(c)+whitespace for c in word])
                pattern = pattern[:-3]

                match = re.findall(pattern, cause)
                if match is not None:
                    cause_template[ner] =  list(match)
                    cause = re.sub(pattern, f'[{ner}]', cause)
                    cause_rem = re.sub(pattern, '[MASK]', cause_rem)

                match = re.findall(pattern, effect)
                if match is not None:
                    effect_template[ner] =  list(match)
                    effect = re.sub(pattern, f'[{ner}]', effect)
                    effect_rem = re.sub(pattern, '[MASK]', effect_rem)

        pos_tag = nltk.pos_tag(nltk.word_tokenize(re.sub('\[MASK\]','',cause_rem)))
        cause_template['ACTION'] = ' '.join([str(word) for word,tag in pos_tag if tag[:2] in ['VB']])
        cause_template['OBJECT'] = ' '.join([str(word) for word,tag in pos_tag if tag[:2] in ['NN']])

        pos_tag = nltk.pos_tag(nltk.word_tokenize(re.sub('\[MASK\]','',effect_rem)))
        effect_template['ACTION'] = ' '.join([str(word) for word,tag in pos_tag if tag[:2] in ['VB']])
        effect_template['OBJECT'] = ' '.join([str(word) for word,tag in pos_tag if tag[:2] in ['NN']])

        cause_action.append(cause)
        effect_action.append(effect)
        cause_action_rem.append(cause_rem)
        effect_action_rem.append(effect_rem)
        evidence_ner.append(tagged)
        cause_store.append(cause_template)
        effect_store.append(effect_template)

    data['cause_action'] = cause_action
    data['effect_action'] = effect_action
    data['cause_action_rem'] = cause_action_rem
    data['effect_action_rem'] = effect_action_rem
    data['evidence_ner'] = evidence_ner
    data['cause_store'] = cause_store
    data['effect_store'] = effect_store
    
    return data


if __name__ == "__main__":   
    graph_folder = r"D:\66 CausalMap\SciLit_CausalMap\visualization\mir_paper"
    get_nodes_infos(graph_folder)