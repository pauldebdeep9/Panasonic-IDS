import os
import sys
import json
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from torch_deps import get_ce_cls, get_ce_span
from stanza_deps import get_dep_pos #, format_sents_for_opensesame, get_frames
from causenet_deps import crawl_nounpairs
from tqdm import tqdm
import logging


def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(levelname)s — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='w+')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def open_json(json_file_path, data_format=list):
    if data_format==dict or data_format=='dict':
        with open(json_file_path) as json_file:
            data = json.load(json_file)
    elif data_format==list or data_format=='list':
        data = []
        for line in open(json_file_path, encoding='utf-8'):
            data.append(json.loads(line))
    else:
        raise NotImplementedError
    return data


# python mass_panasonic.py MarketIntelligenceReport all

v = sys.argv[1] # filename 'MarketIntelligenceReport'
row_id = sys.argv[2] # 0

if row_id=='all':

    batch_size = 15 # n documents to process per time
    start_from = 0 # 0 if from beginning

    file_name = f'data/{v}.csv'
    text_col = pd.read_csv(file_name)['Translated'].iloc[start_from:]

    if batch_size>=len(text_col):

        logger = my_custom_logger(f"outs/{v}.log")
        logger.info('### Splitting sentences...')
        sentences = []
        lengths = []
        for text in text_col:
            sents = sent_tokenize(text)
            lengths.append(len(sents))
            sentences.extend(sents)
        logger.info(f'No. of sents per row: {lengths}')

        logger.info('### Retrieving POS and dependency tags...')
        pos_deps = get_dep_pos(sentences)
        logger.info('\n'+'\n\n'.join(pos_deps))

        logger.info('### Retrieving Causal tags...')
        ce_cls = get_ce_cls(sentences)
        ce_span = get_ce_span(sentences)
        logger.info('\n'+'\n'.join([str(a)+'\t'+str(b) for a,b in zip(ce_cls, ce_span)]))

        logger.info('### Retrieving CauseNet labels...')
        crawl_nounpairs(
            sentences,
            output_dir='outs/', 
            min_support=5,
            demo=None,
            overwrite_output_dir=False, 
            pattern_path_name="causenet_patterns.txt", 
            nounpair_path_name="crawled_nounpairs.txt"
            )
        with open("outs/crawled_nounpairs.txt", 'r') as file:
            nounpairs_txt = file.read()
        logger.info('\n'+nounpairs_txt)

        # logger.info('### Retrieving Frames...')
        # format_sents_for_opensesame(sentences)
        # os.system("sudo ./run_sesame.sh")
        # output_dict = get_frames(sentences)
        # for k,v in output_dict.items():
        #     logger.info(f'sent_id: {k}\n'+'\n'.join(v))

        logger.info('### End of Report...')
    
    else:
        rounds = int(np.ceil(len(text_col)/batch_size))

        for r in tqdm(range(rounds)):
            
            s = r*batch_size
            e = (r+1)*batch_size
            text_col_for_r = text_col[s:e]
            logger = my_custom_logger(f"outs/{v}-{s+start_from}_{e-1+start_from}.log")

            logger.info('### Splitting sentences...')
            sentences = []
            lengths = []
            for text in text_col_for_r:
                sents = sent_tokenize(text)
                lengths.append(len(sents))
                sentences.extend(sents)
            logger.info(f'No. of sents per row: {lengths}')
            
            logger.info('### Retrieving POS and dependency tags...')
            pos_deps = get_dep_pos(sentences)
            logger.info('\n'+'\n\n'.join(pos_deps))

            logger.info('### Retrieving Causal tags...')
            ce_cls = get_ce_cls(sentences)
            ce_span = get_ce_span(sentences)
            logger.info('\n'+'\n'.join([str(a)+'\t'+str(b) for a,b in zip(ce_cls, ce_span)]))

            logger.info('### Retrieving CauseNet labels...')
            crawl_nounpairs(
                sentences,
                output_dir='outs/', 
                min_support=5, 
                demo=None,
                overwrite_output_dir=False, 
                pattern_path_name="causenet_patterns.txt", 
                nounpair_path_name="crawled_nounpairs.txt"
                )
            with open("outs/crawled_nounpairs.txt", 'r') as file:
                nounpairs_txt = file.read()
            logger.info('\n'+nounpairs_txt)

            # logger.info('### Retrieving Frames...')
            # format_sents_for_opensesame(sentences)
            # os.system("sudo ./run_sesame.sh")
            # output_dict = get_frames(sentences)
            # for k,v in output_dict.items():
            #     logger.info(f'sent_id: {k}\n'+'\n'.join(v))

            logger.info('### End of Report...')
        
        # /home/fiona/anaconda3/envs/torchgeom/bin/python mass.py all

else:
    row_id = int(sys.argv[2])
    logger = my_custom_logger(f"outs/{v}-{row_id}.log")
    file_name = f'data/{v}.csv'
    text = pd.read_csv(file_name)['Translated'].iloc[row_id]

    logger.info('### Splitting sentences...')
    sentences = sent_tokenize(text)

    logger.info('### Retrieving POS and dependency tags...')
    pos_deps = get_dep_pos(sentences)
    logger.info('\n'+'\n\n'.join(pos_deps))

    logger.info('### Retrieving Causal tags...')
    ce_cls = get_ce_cls(sentences)
    ce_span = get_ce_span(sentences)
    logger.info('\n'+'\n'.join([str(a)+'\t'+str(b) for a,b in zip(ce_cls, ce_span)]))

    logger.info('### Retrieving CauseNet labels...')
    crawl_nounpairs(
        sentences,
        output_dir='outs/', 
        min_support=5, 
        demo=None,
        overwrite_output_dir=False, 
        pattern_path_name="causenet_patterns.txt", 
        nounpair_path_name="crawled_nounpairs.txt"
        )
    with open("outs/crawled_nounpairs.txt", 'r') as file:
        nounpairs_txt = file.read()
    logger.info('\n'+nounpairs_txt)

    # logger.info('### Retrieving Frames...')
    # format_sents_for_opensesame(sentences)
    # os.system("sudo ./run_sesame.sh")
    # output_dict = get_frames(sentences)
    # for k,v in output_dict.items():
    #     logger.info(f'sent_id: {k}\n'+'\n'.join(v))

    logger.info('### End of Report...')
    # /home/fiona/anaconda3/envs/torchgeom/bin/python mass.py 02