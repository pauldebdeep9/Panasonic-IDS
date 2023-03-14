import re, os, sys
from ast import literal_eval
import pandas as pd
import numpy as np
from copy import copy
import nltk
from nltk.tokenize import sent_tokenize
from nlp_tools import join_neighbouring_args, pick_longer_args, combine_causenet_args
from torch_deps import get_ce_pair_cls


class LogReader(object):
    def __init__(self, log_file_path, csv_file_path, 
                 model, tokenizer, additional_pair_clf=True, 
                 causenet_simplify=False):
        self.doc_id_start, self.doc_id_end = log_file_path.split("MIR-")[-1].split(".log")[0].split("_")
        self.doc_id_start, self.doc_id_end = int(self.doc_id_start), int(self.doc_id_end)
        self.segments = {}
        self.open_log_file(log_file_path)
        self.tokens = []
        self.pos_tags = []
        self.parse_tokens()
        assert len(self.tokens) == len(self.pos_tags)
        
        self.infos = {}
        self.sentences = []
        self.sent_counts = []
        self.parse_sentences(csv_file_path)
        assert len(self.tokens) == len(self.sentences)
        assert len(self.sentences) == sum(self.sent_counts)
        
        self.counter2ids = {}
        self.ids2counter = {}
        self.additional_pair_clf = additional_pair_clf
        self.model = model
        self.tokenizer = tokenizer
        self.parse_unicausal()
        
        self.causenet_simplify = causenet_simplify
        self.parse_causenet()
    
    
    def open_log_file(self, log_file_path):
        # open file
        with open(log_file_path) as f:
            lines = f.read().splitlines()
        
        # parse txt to dict
        segment_titles = "— INFO — ###"
        log_info = "— INFO —"
        title = None

        for line in lines:
            if segment_titles in line:
                title = line.split(segment_titles)[-1].strip()
                self.segments[title] = []
            else:
                self.segments[title].append(line)
        
#         self.sent_counts = literal_eval(
#             self.segments['Splitting sentences...'][0].split('No. of sents per row: ')[-1]
#         )
        

    def parse_sentences(self, csv_file_path):
        text_col = pd.read_csv(csv_file_path)['Translated'].iloc[self.doc_id_start:self.doc_id_end+1]
        for text in text_col:
            text = re.sub('\u200b','',text)
            sents = sent_tokenize(text)
            self.sent_counts.append(len(sents))
            self.sentences.extend(sents)
        
    
    def parse_tokens(self):
        tokens = []
        pos_tags = []
        for line in self.segments['Retrieving POS and dependency tags...'][1:]:
            if line.strip()=='':
                # new sentence
                self.tokens.append(tokens)
                self.pos_tags.append(pos_tags)
                tokens=[]
                pos_tags = []
            else:
                # continue sentence
                line = line.split('\t')
                tokens.append(line[1].split('word: ')[-1])
                pos_tags.append(line[2].split('pos: ')[-1])
        # last round
        self.tokens.append(tokens)
        self.pos_tags.append(pos_tags)
    
    
    def parse_unicausal(self):
        doc_id = 0
        doc_last_sent_id = self.sent_counts[doc_id] 
        template = {'unicausal': None, 'causenet': None, 'unicausal+':None}
        
        for counter, line in enumerate(self.segments['Retrieving Causal tags...'][1:]):
            if counter>=doc_last_sent_id:
                doc_id+=1
                doc_last_sent_id+=self.sent_counts[doc_id] 
            sent_id = counter-(doc_last_sent_id-self.sent_counts[doc_id])
            self.counter2ids[counter] = (doc_id, sent_id)

            if doc_id not in self.infos.keys():
                self.infos[doc_id] = {}
            if sent_id not in self.infos[doc_id].keys():
                self.infos[doc_id][sent_id] = copy(template)

            line = line.split('\t')
#             self.sentences.append(re.sub(' ##','',clean_tok(line[1])))

            if int(line[0])==1: # causal
                text = join_neighbouring_args(line[1])
                text = pick_longer_args(text)
                text = re.sub('##','',text)
                text = re.sub('\u200b','',text)
                causes = re.findall(r'<ARG0>(.*?)</ARG0>', text)
                effects = re.findall(r'<ARG1>(.*?)</ARG1>', text)
                
                # get all possible combinations
                relations = []
                for cause in causes:
                    for effect in effects:
                        relations.append([cause,effect])
                
                # insert joined options
                if len(causes)>1 or len(effects)>1:
                    causes+=[''.join(l) for l in re.findall(r'<ARG0>(.*?)</ARG0>(.*?)<ARG1>(.*?)</ARG1>(.*?)<ARG0>(.*?)</ARG0>', text)]
                    effects+=[''.join(l) for l in re.findall(r'<ARG1>(.*?)</ARG1>(.*?)<ARG0>(.*?)</ARG0>(.*?)<ARG1>(.*?)</ARG1>', text)]

                self.infos[doc_id][sent_id]['unicausal'] = {
                    'n_rels': len(relations),
                    'rels': relations,
                    'causes': causes,
                    'effects': effects
                }
            else: # non-causal
                if self.additional_pair_clf:
                    # extract the "clean" examples, verify through Pair Classification
                    text = re.sub('##','',line[1])
                    text = re.sub('\u200b','',text)
                    causes = re.findall(r'<ARG0>(.*?)</ARG0>', text)
                    effects = re.findall(r'<ARG1>(.*?)</ARG1>', text)
                    if len(causes)==1 and len(effects)==1:
                        ce_cls = get_ce_pair_cls(self.model,self.tokenizer,[line[1]])[0]
                        if int(ce_cls)==1:
                            self.infos[doc_id][sent_id]['unicausal+'] = {
                                'n_rels': 1,
                                'rels': [[causes[0],effects[0]]]
                            }

        self.ids2counter = {v:k for k,v in self.counter2ids.items()}
        
    
    def parse_causenet(self):
        prev_text = None
        tmp_ids = []
        
        for line in self.segments['Retrieving CauseNet labels...'][1:-1]:
            line = line.split(';;')
            cause, effect, cause_id, effect_id, pattern = line[0:5]
            text = re.sub('[^A-Za-z0-9]+','',line[-2])
#             print('>>>',text)
            if prev_text!=text:
                counter = 0
                sent_scan = re.sub('[^A-Za-z0-9]+','',self.sentences[counter])
                while text not in sent_scan:
#                     print(':::',sent_scan)
                    counter+=1
                    sent_scan = re.sub('[^A-Za-z0-9]+','',self.sentences[counter])
                doc_id, sent_id = self.counter2ids[counter]
                
            if self.infos[doc_id][sent_id]['causenet'] is None:
                self.infos[doc_id][sent_id]['causenet'] = {
                    'n_rels': 1,
                    'rels': [[cause,effect]], 
                    'patterns': [pattern],
                    'rels_id': [[cause_id,effect_id]]
                }
            else:
                self.infos[doc_id][sent_id]['causenet']['n_rels'] += 1
                self.infos[doc_id][sent_id]['causenet']['rels'].append([cause,effect])
                self.infos[doc_id][sent_id]['causenet']['patterns'].append(pattern)
                self.infos[doc_id][sent_id]['causenet']['rels_id'].append([cause_id,effect_id])
            tmp_ids.append((doc_id, sent_id))
            prev_text = text
            
        if self.causenet_simplify:
            # join up args that relate to the same pattern and cause/effect
            for doc_id, sent_id in set(tmp_ids):
                ddict = self.infos[doc_id][sent_id]['causenet']
                tokens = self.tokens[self.ids2counter[(doc_id,sent_id)]]
                self.infos[doc_id][sent_id]['causenet'] = combine_causenet_args(ddict, tokens)


if __name__ == "__main__":
    # Unit Tests
    log_file_path = r"D:\79 Mass\outs\MIR\MIR-4125_4139.log"
    csv_file_path = r"D:\66 CausalMap\Panasonic-IDS\data\MIR.csv"
    print(log_file_path)
    reader = LogReader(
        log_file_path, 
        csv_file_path, 
        causenet_simplify=True
        )
    print(reader.infos)