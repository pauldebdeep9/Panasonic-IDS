from torch_deps import get_ce_pair_cls
from nlp_tools import get_text_w_pairs, get_pos_tags
import itertools


def insert_into_finfos(f_infos, keys, values):
    if keys not in f_infos.keys():
        f_infos[keys] = [values]
    else:
        f_infos[keys].append(values)
    return f_infos


def get_ce_spans_for_causenet(patterns, rel_ids, tokens, pos_tags):
    signals = [p.split('/') for p in patterns.split('\t') if ('/' in p) and ('[[cause]]' not in p) and ('[[effect]]' not in p)]
    cause_ids, effect_ids = rel_ids
    cause_ids = [int(i) for i in cause_ids.split('_')]
    effect_ids = [int(i) for i in effect_ids.split('_')]
    start_ids = min(cause_ids+effect_ids) # inclusive, 1-indexed
    end_ids = max(cause_ids+effect_ids) # inclusive, 1-indexed
    
    # print((start_ids, end_ids), (cause_ids, effect_ids), signals)
    # print(tokens[start_ids-1:end_ids])
    # print(pos_tags[start_ids-1:end_ids])
    
    signal_ids = []
    for (signal_word, signal_pos) in signals:
        for j in range(start_ids-1,end_ids):
            if (signal_word.lower()==tokens[j]) and (signal_pos==pos_tags[j]):
                signal_ids.append(j+1) # change to 1-indexed
    # print(signal_ids)
    
    cause_start_ids = min(cause_ids+[i+1 for i in signal_ids]) # change signal to 1 after so that it's excluding signal
    cause_end_ids = max(cause_ids+[i-1 for i in signal_ids]) # change signal to 1 before so that it's excluding signal
    cause_span = tokens[cause_start_ids-1:cause_end_ids]

    effect_start_ids = min(effect_ids+[i+1 for i in signal_ids]) # change signal to 1 after so that it's excluding signal
    effect_end_ids = max(effect_ids+[i-1 for i in signal_ids]) # change signal to 1 before so that it's excluding signal
    effect_span = tokens[effect_start_ids-1:effect_end_ids]
    
    return ' '.join(cause_span), ' '.join(effect_span)


def retain_intersection(reader, f_infos:dict={}, doc_id_start:int=0,
                        model=None, tokenizer=None, additional_pair_clf=True):
    # Retain examples with both unicausal and causenet
    
    for doc_id in reader.infos.keys():
        for sent_id in reader.infos[doc_id].keys():
            
            # print(doc_id, sent_id)
            # print(reader.infos[doc_id][sent_id])
            
            unicausal_dict = reader.infos[doc_id][sent_id]['unicausal']
            unicausalp_dict = reader.infos[doc_id][sent_id]['unicausal+']
            causenet_dict = reader.infos[doc_id][sent_id]['causenet']
    
            # Causal relation is identified in BOTH UniCausal and CauseNet
            if (unicausal_dict is not None) and (causenet_dict is not None):
                remove_idx = []
                remove_ids = []
                for idx, (cause_span, effect_span) in enumerate(unicausal_dict['rels']):
                    for ids, (cause, effect) in enumerate(causenet_dict['rels']):
                        if (cause in cause_span) and (effect in effect_span):
                            e_dict = {
                                'text': reader.sentences[reader.ids2counter[(doc_id,sent_id)]], # string of text
                                'cause_span': cause_span, # long cause span
                                'effect_span': effect_span, # long effect span
                                'doc_id': int(doc_id)+int(doc_id_start),
                                'sent_id': sent_id,
                                'method': ['unicausal','causenet'],
                                'causenet_patterns': causenet_dict['patterns'][ids]
                            }
                            f_infos = insert_into_finfos(f_infos, keys=(cause_span,effect_span), values=e_dict)
                            remove_idx.append(idx)
                            remove_ids.append(ids)
                # Remove appended items to avoid duplicates later
                unicausal_dict['rels'] = [r for i,r in enumerate(unicausal_dict['rels']) if i not in remove_idx]
                causenet_dict['rels'] = [r for i,r in enumerate(causenet_dict['rels']) if i not in remove_ids]
                causenet_dict['rels_id'] = [r for i,r in enumerate(causenet_dict['rels_id']) if i not in remove_ids]
                causenet_dict['patterns'] = [r for i,r in enumerate(causenet_dict['patterns']) if i not in remove_ids]
            
            # Handling causal relations identified by EITHER
            if unicausal_dict is not None:
                if len(unicausal_dict['rels'])==1:
                    # Only keep examples that have ONE causal relation
                    # Because current UniCausal is not meant for multiple causal relation extraction
                    cause_span, effect_span = unicausal_dict['rels'][0]
                    e_dict = {
                        'text': reader.sentences[reader.ids2counter[(doc_id,sent_id)]],
                        'cause_span': cause_span,
                        'effect_span': effect_span,
                        'doc_id': int(doc_id)+int(doc_id_start),
                        'sent_id': sent_id,
                        'method': ['unicausal'],
                        'causenet_patterns': None
                    }
                    f_infos = insert_into_finfos(f_infos, keys=(cause_span,effect_span), values=e_dict)
                elif len(unicausal_dict['rels'])>1:

                    if additional_pair_clf:
                        text = reader.sentences[reader.ids2counter[(doc_id,sent_id)]]
                        sentences = []
                        arguments = []
                        for span in unicausal_dict['args']:
                            if 'NN' in get_pos_tags(span):
                                arguments.append(span)
                        pairs1 = list(itertools.combinations(arguments, 2))
                        pairs2 = [(two,one) for one,two in pairs1]
                        arg_pairs = pairs1+pairs2

                        for cause_span, effect_span in arg_pairs:
                            text_w_pairs = get_text_w_pairs(text, cause_span, effect_span)
                            sentences.append(text_w_pairs)
                        ce_cls = get_ce_pair_cls(model, tokenizer, sentences)

                        for i, pred in enumerate(ce_cls):
                            if int(pred)==1:
                                if ('<ARG0>' in sentences[i]) and ('<ARG1>' in sentences[i]):
                                    cause_span, effect_span = arg_pairs[i]
                                    e_dict = {
                                        'text': text,
                                        'cause_span': cause_span,
                                        'effect_span': effect_span,
                                        'doc_id': int(doc_id)+int(doc_id_start),
                                        'sent_id': sent_id,
                                        'method': ['unicausalm'],
                                        'causenet_patterns': None
                                    }
                                    f_infos = insert_into_finfos(f_infos, keys=(cause_span,effect_span), values=e_dict)
            
            if unicausalp_dict is not None:
                cause_span, effect_span = unicausalp_dict['rels'][0]
                e_dict = {
                    'text': reader.sentences[reader.ids2counter[(doc_id,sent_id)]],
                    'cause_span': cause_span,
                    'effect_span': effect_span,
                    'doc_id': int(doc_id)+int(doc_id_start),
                    'sent_id': sent_id,
                    'method': ['unicausal+'],
                    'causenet_patterns': None
                }
                f_infos = insert_into_finfos(f_infos, keys=(cause_span,effect_span), values=e_dict)

            if causenet_dict is not None:
                for ids in range(len(causenet_dict['rels_id'])): # loop over each relation
                    cause_span, effect_span = get_ce_spans_for_causenet(
                        patterns = causenet_dict['patterns'][ids], 
                        rel_ids = causenet_dict['rels_id'][ids],
                        tokens = reader.tokens[reader.ids2counter[(doc_id,sent_id)]], 
                        pos_tags = reader.pos_tags[reader.ids2counter[(doc_id,sent_id)]]
                    )
                    cause, effect = causenet_dict['rels'][ids]
                    e_dict = {
                        'text': reader.sentences[reader.ids2counter[(doc_id,sent_id)]],
                        'cause_span': cause_span,
                        'effect_span': effect_span,
                        'doc_id': int(doc_id)+int(doc_id_start),
                        'sent_id': sent_id,
                        'method': ['causenet'],
                        'causenet_patterns': causenet_dict['patterns'][ids]
                    }
                    f_infos = insert_into_finfos(f_infos, keys=(cause_span,effect_span), values=e_dict)
            
