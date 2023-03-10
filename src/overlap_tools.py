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


def retain_intersection(reader, f_infos:dict={}, doc_id_start:int=0):
    # Retain examples with both unicausal and causenet
    
    for doc_id in reader.infos.keys():
        for sent_id in reader.infos[doc_id].keys():
            
            # print(doc_id, sent_id)
            # print(reader.infos[doc_id][sent_id])
            
            unicausal_dict = reader.infos[doc_id][sent_id]['unicausal']
            causenet_dict = reader.infos[doc_id][sent_id]['causenet']
    
            # Causal relation is identified in BOTH UniCausal and CauseNet
            if (unicausal_dict is not None) and (causenet_dict is not None):
                for cause_span, effect_span in unicausal_dict['rels']:
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
                            if (cause_span,effect_span) not in f_infos.keys():
                                f_infos[(cause_span,effect_span)] = [e_dict]
                            else:
                                f_infos[(cause_span,effect_span)].append(e_dict)
            # Handling causal relations identified by EITHER
            elif unicausal_dict is not None:
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
                    if (cause_span,effect_span) not in f_infos.keys():
                        f_infos[(cause_span,effect_span)] = [e_dict]
                    else:
                        f_infos[(cause_span,effect_span)].append(e_dict)
            elif causenet_dict is not None:
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
                    if (cause_span,effect_span) not in f_infos.keys():
                        f_infos[(cause_span,effect_span)] = [e_dict]
                    else:
                        f_infos[(cause_span,effect_span)].append(e_dict)
                         