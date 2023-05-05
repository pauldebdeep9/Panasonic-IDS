# -*- coding: utf-8 -*-
import os
import torch
import stanza
import logging
from typing import Union
from tqdm import tqdm
import networkx as nx
from collections import Counter
logger = logging.getLogger(__name__)

device = 'gpu' if torch.cuda.is_available() else 'cpu'
stanford_nlp = stanza.Pipeline(
    processors='tokenize,mwt,pos,lemma,depparse',
    lang='en',
    use_gpu=device
    )

_CITATION = """\
@inproceedings{heindorf2020causenet,
  author    = {Stefan Heindorf and
               Yan Scholten and
               Henning Wachsmuth and
               Axel-Cyrille Ngonga Ngomo and
               Martin Potthast},
  title     = CauseNet: Towards a Causality Graph Extracted from the Web,
  booktitle = CIKM,
  publisher = ACM,
  year      = 2020
}
"""

_DESCRIPTION = """\
Crawled Wikipedia Data from CIKM 2020 paper 
'CauseNet: Towards a Causality Graph Extracted from the Web.' 
"""
_URL = "https://github.com/causenet-org/CIKM-20"


# BEFOREHAND:

# 1. EXTRACT PATTERNS from CauseNet using "62 CausalTM" (incomplete)
# 105220 patterns in total, 2314 patterns with >=5 support

# 2. FORMAT PATTERNS into dict using code:
# http://localhost:8888/notebooks/79%20Mass/notebooks/Get%20CauseNet%20Pattern%20Dict.ipynb

class CauseNetPatternCrawler(object): 
    """
    Given seed causal pairs, we extract linguistic patterns.
    This code is a reimplementation of CauseNet (originally written in Java).
    There are some discrepancies in the implementation of pattern extraction and 
    especially pertaining to dep parsing from Stanza versions. However, the logic
    and idea of setting seed causal noun pairs to get linguistic patterns filtered
    by a minimum support to find more causal noun pairs is the same.
    
    """
    def __init__(self, min_support=5, output_dir='outs/', patterns=None, overwrite_output_dir=True, \
        delimiter=';;', device='cpu', verbose=False, pattern_path_name="crawled_patterns.txt", \
        nounpair_path_name="crawled_nounpairs.txt"):
        # To do: Rewrite into object
        # Functions: show_example(id), extract(),
        self.min_support = min_support
        self.pattern_path = os.path.join(output_dir, pattern_path_name)
        self.nounpair_path = os.path.join(output_dir, nounpair_path_name)
        
        if patterns is None:
            if os.path.isfile(self.pattern_path):
                if overwrite_output_dir:
                    self.patterns = []
                else:
                    ddict = {}
                    with open(self.pattern_path) as f:
                        for line in f:
                            line = line.split(delimiter)
                            ddict[str(line[0])] = int(line[1])
                    self.patterns = Counter(ddict)
                    logger.info(f'Loaded pre-extracted patterns from "{self.pattern_path}"')
            else:
                self.patterns = []
        else:
            self.patterns = patterns
        self.stanford_nlp = stanford_nlp
        self.delimiter = delimiter
        self.causal_patterns = None
        self.verbose = verbose
    

    # def pointer(self, example:dict):
    #     """
    #     Default pointer (based on CauseNetWiki)
    #     Users can amend this function accordingly to their dataset.
    #     """
    #     # point to text
    #     text = example['sentence']
    #     # point to seeds
    #     if 'cause_word' in example.keys():
    #         cause_word = example['cause_word']
    #     else:
    #         cause_word = ''
    #     if 'effect_word' in example.keys():
    #         effect_word = example['effect_word']
    #     else:
    #         effect_word = ''
    #     return text, cause_word, effect_word


    # def extract_patterns(self, examples):
    #     # Iterate over all examples
    #     patterns = []
    #     for ix, example in enumerate(tqdm(examples)):
    #         patterns.extend(self.get_patterns(self.pointer(example)))

    #     # Clean up
    #     patterns = self.clean_patterns(patterns)

    #     # Count & Sort
    #     patterns_count = Counter(patterns)

    #     # Save patterns
    #     self.save_patterns(patterns_count)

    
    def filter_patterns(self):
        if len(self.patterns)==0:
            raise ValueError('Patterns need to be initialised!')
        else:
            causal_patterns = [k for k, c in self.patterns.items() if c >= self.min_support]
            print(f'Working with n={len(causal_patterns)} causal patterns...')
            return causal_patterns
    

    # def extract_nounpairs(self, examples):
        
    #     causal_patterns = self.filter_patterns()

    #     # Iterate over all examples
    #     nounpairs = []
    #     for ix, example in enumerate(tqdm(examples)):
    #         text, _, _ = self.pointer(example)
    #         nounpairs.extend(self.match_patterns(text, causal_patterns))

    #     # Count & Sort
    #     nounpairs_count = Counter(nounpairs)

    #     # Save
    #     self.save_nounpairs(nounpairs_count)


    def save_patterns(self, patterns_count):
        # Save patterns
        with open(self.pattern_path, 'w') as fp:
            for line in patterns_count.most_common():
                # Pattern ; Count
                fp.write(str(line[0])+self.delimiter+str(line[1])+'\n')
        self.patterns = patterns_count
    

    def save_nounpairs(self, nounpairs_count):
        # Save patterns
        with open(self.nounpair_path, 'w') as fp:
            for line in nounpairs_count.most_common():
                # Cause ; Effect ; Pattern ; Count
                (c, e, c_id, e_id, pt, st, en, st_id, en_id, ev), count = line
                fp.write(str(c)+self.delimiter+str(e)+self.delimiter+\
                    str(c_id)+self.delimiter+str(e_id)+self.delimiter+str(pt)+self.delimiter+\
                    str(st)+self.delimiter+str(en)+self.delimiter+\
                    str(st_id)+self.delimiter+str(en_id)+self.delimiter+str(ev)+self.delimiter+\
                    str(count)+'\n')


    def get_patterns(self, parsed):
        # Get info
        text, cause_word, effect_word = parsed
        doc = self.stanford_nlp(text)
        graph = self.plot_doc_graph(doc)
        
        # Initialise
        tokens = {}
        prev_count = 0
        cause_ids = []
        effect_ids = []
        # Check if cause or effect is modified by 'case'
        for sent in doc.sentences:
            for tok in sent.words:
                tokens[tok.id+prev_count]= {
                    'text': tok.text,
                    'head': tok.head+prev_count,
                    'xpos': tok.xpos,
                    'deprel': tok.deprel
                }
                if tok.text==cause_word:
                    cause_ids.append(tok.id+prev_count)
                if tok.text==effect_word:
                    effect_ids.append(tok.id+prev_count)
            # for multiple sentences
            prev_count+=tok.id

        patterns = []
        signal_paths = []
        for cause_id in cause_ids:
            for effect_id in effect_ids:
                _patterns, _signal_paths = self.find_path(graph, tokens, cause_id, effect_id)
                patterns.extend(_patterns)
                signal_paths.extend(_signal_paths)
        
        return patterns


    def find_path(self, graph, tokens, cause_id, effect_id):
        patterns = []
        signal_paths = []

        for k,v in tokens.items():
            if v['deprel']=='case':
                if v['head']==cause_id:
                    _tok = tokens[cause_id]
                    _tok['deprel'] = '-nmod:'+ str(v['text'])
                    tokens[cause_id] = _tok
                elif v['head']==effect_id:
                    _tok = tokens[effect_id]
                    _tok['deprel'] = '+nmod:'+ str(v['text'])
                    tokens[effect_id] = _tok

        for path in nx.all_simple_paths(graph, source=cause_id, target=effect_id, cutoff=8):

            dep_chain = ''
            path_of_word_ids = []

            for i,j in enumerate(path):

                # get edge
                if i==0:
                    # first item
                    pass
                else:
                    if 'nmod:' in tokens[path[i-1]]['deprel']:
                        # previous
                        dep_chain+= str(tokens[path[i-1]]['deprel'])+'\t'
                    elif 'nmod:' in tokens[j]['deprel']:
                        # current
                        dep_chain+= str(tokens[j]['deprel'])+'\t'
                    elif (j,path[i-1]) in graph.edges:
                        # (head, tail) : (current, previous)
                        if tokens[j]['deprel']=='root':
                            dep_chain+='-'+str(tokens[path[i-1]]['deprel'])+'\t'
                        else:
                            dep_chain+='+'+str(tokens[j]['deprel'])+'\t'

                    elif (path[i-1],j) in graph.edges:
                        # (head, tail) : (previous, current)
                        if tokens[path[i-1]]['deprel']=='root':
                            dep_chain+='+'+str(tokens[j]['deprel'])+'\t'
                        else:
                            dep_chain+='-'+str(tokens[path[i-1]]['deprel'])+'\t'
                    else:
                        raise UnboundLocalError('Not possible! Check code.')

                # get node
                if j==cause_id:
                    dep_chain+='[[cause]]/N\t'
                elif j==effect_id:
                    dep_chain+='[[effect]]/N\t'
                else:
                    dep_chain+=str(tokens[j]['text'])+'/'+str(tokens[j]['xpos'])+'\t'
                    signal_word_ids.append(j)
            patterns.append(dep_chain)
            signal_paths.apppend(signal_word_ids)
        
        return patterns, signal_paths


    def clean_patterns(self, patterns):
        # clean up NP chains
        all_patterns = []
        for p in patterns:
            pt = p.split('\t')
            length = len(pt)
            remove = []
            
            for i in range(2,length-1,2):
                if any([non in pt[i] for non in ['NN', 'CD']]):
                    remove.append(i)
                else:
                    break
            
            for i in range(length-4,2,-2):
                if any([non in pt[i] for non in ['NN', 'CD']]):
                    remove.append(i)
                else:
                    break
            
            keep = []
            if len(set(remove))==len(remove):
                # not whole phrase is NOUNS
                for i in range(0,length-1,2):
                    if i not in remove:
                        keep.append(pt[i])
                        keep.append(pt[i+1])
            
            length = len(keep)
            keep = '\t'.join(keep)
            # drop empty chains
            # '[[cause]]/N\t-nmod:of\t[[effect]]/N\t'
            # drop direct NP1-> NP2 relations (Min NP1->DEP1->REL->DEP2->NP2)
            if keep!='' and length>=5:
                all_patterns.append(keep.strip())
                
        return all_patterns


    def plot_doc_graph(self, doc, verbose=False):
        """
        For each document/sentence, we can plot the dependency graph from Stanza dependency parser.
        Orange nodes represent Cause; Green nodes represent Effect; Grey nodes represent Others.
        """
        
        G = nx.DiGraph() # directed graph
        prev_count = 0
        for sent in doc.sentences:

            for word in sent.words:
                G.add_nodes_from([(prev_count+word.id, {'text': word.text})])
                if word.head>0:
                    G.add_edge(word.head+prev_count, word.id+prev_count)
                
                if self.verbose:
                    print(f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: '+\
                        f'{sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
                        )
            prev_count+=word.id

        return G.to_undirected()


    def show_example(self, save_graph=False):
        pass


    def match_patterns(self, text:str, causal_patterns:Union[Counter,list,None]=None):
        # If none
        if causal_patterns is None:
            if self.causal_patterns is None:
                raise ValueError('causal_patterns has not been defined!')
            else:
                causal_patterns = self.causal_patterns
        
        # Get info
        doc = self.stanford_nlp(text)
        graph = self.plot_doc_graph(doc)
        
        # Initialise
        tokens = {}
        sent2tid = {0:0}
        prev_count = 0
        noun_ids = []
        # Check if cause or effect is modified by 'case'
        for sent_id, sent in enumerate(doc.sentences):
            for tok in sent.words:
                tokens[tok.id+prev_count]= {
                    'text': tok.text,
                    'head': tok.head+prev_count,
                    'xpos': tok.xpos,
                    'deprel': tok.deprel,
                    'sent_id': sent_id+1
                }
                if tok.xpos[0:2]=='NN':
                    noun_ids.append(tok.id+prev_count)
            # for multiple sentences
            prev_count+=tok.id
            # save token id of each end of sentence
            sent2tid[sent_id+1]=prev_count

        if self.verbose:
            print(tokens)
        
        nounpairs = []

        for cause_id in noun_ids:
            for effect_id in noun_ids:
                if cause_id==effect_id:
                    # not the same word
                    continue

                patterns, signal_paths = self.find_path(graph, tokens, cause_id, effect_id)
                patterns = self.clean_patterns(patterns)

                if self.verbose:
                    print(f'{cause_id}, {effect_id}: {patterns}')

                matched = []
                evidence = []
                for pt in patterns:
                    if pt in causal_patterns:
                        matched.append(pt)
                        start_sent_id=min(tokens[cause_id]['sent_id'],tokens[effect_id]['sent_id'])
                        end_sent_id=max(tokens[cause_id]['sent_id'],tokens[effect_id]['sent_id'])
                        start=sent2tid[start_sent_id-1]+1
                        end=sent2tid[end_sent_id]+1
                        evidence.append(' '.join([tokens[i]['text'] for i in range(start,end)]))
                
                if len(matched)>0:
                    cause = tokens[cause_id]['text']
                    effect = tokens[effect_id]['text']
                    nounpairs.append((
                        cause, effect, cause_id, effect_id, '//'.join(matched),
                        start_sent_id, end_sent_id, start, end, '//'.join(evidence)
                        ))
        return nounpairs


def crawl_nounpairs(examples, output_dir='outs/', min_support=5, demo=50, overwrite_output_dir=False, \
    pattern_path_name="crawled_patterns.txt", nounpair_path_name="crawled_nounpairs.txt"):

    # Load data
    trainer = CauseNetPatternCrawler(
        min_support=min_support,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        device=device,
        pattern_path_name=pattern_path_name,
        nounpair_path_name=nounpair_path_name
    )

    causal_patterns = trainer.patterns
    causal_patterns = [k for k,c in causal_patterns.items() if c >= trainer.min_support]
    logger.info(f'Working with n={len(causal_patterns)} causal patterns...')

    nounpairs = []
    # Iterate over all examples
    for ix, text in enumerate(tqdm(examples)):
        if demo is not None:
            if ix>int(demo):
                break
        nounpairs.extend(trainer.match_patterns(text, causal_patterns))

    # Count & Sort
    nounpairs_count = Counter(nounpairs)
    print(nounpairs_count)

    # Save
    trainer.save_nounpairs(nounpairs_count)

    # # Format
    # outputs = ''
    # for line in nounpairs_count.most_common():
    #     # Cause ; Effect ; Pattern ; Count
    #     (c, e, c_id, e_id, pt, st, en, st_id, en_id, ev), count = line
    #     outputs += str(c)+self.delimiter+str(e)+self.delimiter+\
    #         str(c_id)+self.delimiter+str(e_id)+self.delimiter+str(pt)+self.delimiter+\
    #         str(st)+self.delimiter+str(en)+self.delimiter+\
    #         str(st_id)+self.delimiter+str(en_id)+self.delimiter+str(ev)+self.delimiter+\
    #         str(count)+'\n'

    # return outputs
 