import re
import nltk


def get_pos_tags(text):
    text = nltk.word_tokenize(text)
    return [pos for word,pos in nltk.pos_tag(text)]


def join_neighbouring_args(text):
    """
    The government said in a statement <ARG0>the incentive scheme</ARG0> was expected <ARG0>to help</ARG0> ...
    -->
    The government said in a statement <ARG0>the incentive scheme was expected to help</ARG0> ...
    
    """
    patterns = re.findall(r'</ARG0>(.*?)<ARG0>', text)
    patterns = [p for p in patterns if 'ARG1' not in p]
    for p in patterns:
        text = re.sub(re.escape(f'</ARG0>{p}<ARG0>'), p, text)
    
    patterns = re.findall(r'</ARG1>(.*?)<ARG1>', text)
    patterns = [p for p in patterns if 'ARG0' not in p]
    for p in patterns:
        text = re.sub(re.escape(f'</ARG1>{p}<ARG1>'), p, text)
    
    return text


def pick_longer_args(text):
    """
    In cases where we have 2 Causes 1 Effect or 1 Cause 2 Effect, we simplify to 1 Cause 1 Effect.
    <ARG1>"</ARG1> Due to <ARG0>the current situation in this region</ARG0> <ARG1>, there may be disruptions in the supply chain . "</ARG1>
    -->
    Due to <ARG0>the current situation in this region</ARG0> <ARG1>, there may be disruptions in the supply chain . "</ARG1>
    
    """
    
    cause_patterns = re.findall(r'<ARG0>(.*?)</ARG0>', text)
    effect_patterns = re.findall(r'<ARG1>(.*?)</ARG1>', text)
    if len(cause_patterns)==2 and len(effect_patterns)==1:
        if len(cause_patterns[0])>=len(cause_patterns[1]):
            # keep first, remove second
            p = cause_patterns[1]
        else:
            # keep second, remove first
            p = cause_patterns[0]
        text = re.sub(re.escape(f'<ARG0>{p}</ARG0>'), p, text)
    elif len(effect_patterns)==2 and len(cause_patterns)==1:
        if len(effect_patterns[0])>=len(effect_patterns[1]):
            # keep first, remove second
            p = effect_patterns[1]
        else:
            # keep second, remove first
            p = effect_patterns[0]
        text = re.sub(re.escape(f'<ARG1>{p}</ARG1>'), p, text)
    else:
        # do nothing
        pass
    return text


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub('</*[A-Z]+\d*>','',tok) 


def simplify_chain(chain):
    return list(str(i) for i in sorted(set([int(i) for i in chain])))
    
    
def get_words(chain, tokens):
    ss = int(chain[0])-1 # 1-index to 0-index
    ee = int(chain[-1])-1 # 1-index to 0-index
    return ' '.join(tokens[ss:ee+1])


def combine_causenet_args(ddict, tokens):
    updated_ddict = {'patterns':[], 'rels':[], 'rels_id':[]}
    relations = []
    
    prev_pattern = None
    prev_cause = None
    prev_effect = None
    same_cause_or_effect = None
    cause_chain = []
    effect_chain = []

    def reset(current_cause, current_effect):
        same_cause_or_effect = None
        cause_chain = [current_cause]
        effect_chain = [current_effect]
        
    def differing(prev_pattern, cause_chain, effect_chain):
        # differing series already
        updated_ddict['patterns'].append(prev_pattern)
        cause_chain = simplify_chain(cause_chain)
        effect_chain = simplify_chain(effect_chain)
        updated_ddict['rels_id'].append(['_'.join(cause_chain), '_'.join(effect_chain)])
        updated_ddict['rels'].append([get_words(cause_chain, tokens), get_words(effect_chain, tokens)])

          
    for r_id in range(ddict['n_rels']):

        current_pattern = str(ddict['patterns'][r_id])
        current_cause, current_effect = ddict['rels_id'][r_id]

        if current_pattern==prev_pattern:
            if current_cause==prev_cause:
                if same_cause_or_effect=='effect':
                    differing(prev_pattern, cause_chain, effect_chain)
                    reset(current_cause, current_effect)
                else:
                    same_cause_or_effect='cause'
                    cause_chain.append(current_cause)
                    effect_chain.append(current_effect)
            elif current_effect==prev_effect:
                if same_cause_or_effect=='cause':
                    differing(prev_pattern, cause_chain, effect_chain)
                    reset(current_cause, current_effect)
                else:
                    same_cause_or_effect='effect'
                    cause_chain.append(current_cause)
                    effect_chain.append(current_effect)
            else:
                differing(prev_pattern, cause_chain, effect_chain)
                reset(current_cause, current_effect)
        else:
            if prev_pattern is not None:
                differing(prev_pattern, cause_chain, effect_chain)
            cause_chain = [current_cause]
            effect_chain = [current_effect]
        
        prev_pattern=current_pattern
        prev_cause=current_cause
        prev_effect=current_effect
    
    # last round append
    differing(prev_pattern, cause_chain, effect_chain)
    updated_ddict['n_rels'] = len(updated_ddict['patterns'])
    
    return updated_ddict


def find_substring_in_text(text, substring):
    # replace [UNK] with specials
    substring = re.sub('\[UNK\]','\u200b',substring)
    # remove all spaces
    substring = re.sub(' ','',substring)    
    # insert optional spaces
    new_substring = ''
    for i in substring:
        # remember to escape chars like () \ etc.
        new_substring+=re.escape(i)+'\s*'
    new_substring = re.sub('\u200b','.*',new_substring)
    search = re.search(new_substring,text)
    
    if search is None:
        raise ValueError(f"{text}\n{substring}")
    
    return search.start(), search.end()


def get_text_w_pairs(text,cause,effect,allow_overlap=False):

    cause_start, cause_end = find_substring_in_text(text, cause)
    effect_start, effect_end = find_substring_in_text(text, effect)
    
    text_w_pairs = text
    if cause_start<effect_start: 
        if effect_start<cause_end:    # <ARG0><ARG1></ARG0>
            if not allow_overlap:
                return None
            effect_start+=6
        else:                         # <ARG0></ARG0><ARG1>
            effect_start+=6+7
        if effect_end<cause_end:      # <ARG0></ARG1></ARG0>
            if not allow_overlap:
                return None
            effect_end+=6
        else:                         # <ARG0></ARG0></ARG1>
            effect_end+=6+7
        text_w_pairs = text_w_pairs[:cause_start] + '<ARG0>' + text_w_pairs[cause_start:cause_end] + '</ARG0>'+ text_w_pairs[cause_end:]
        text_w_pairs = text_w_pairs[:effect_start] + '<ARG1>' + text_w_pairs[effect_start:effect_end] + '</ARG1>'+ text_w_pairs[effect_end:]
    else:
        if cause_start<effect_end:    # <ARG1><ARG0></ARG1>
            if not allow_overlap:
                return None
            cause_start+=6
        else:                         # <ARG1></ARG1><ARG0>
            cause_start+=6+7
        if cause_end<effect_end:      # <ARG1></ARG0></ARG1>
            if not allow_overlap:
                return None
            cause_end+=6
        else:                         # <ARG1></ARG1></ARG0>
            cause_end+=6+7
        text_w_pairs = text_w_pairs[:effect_start] + '<ARG1>' + text_w_pairs[effect_start:effect_end] + '</ARG1>'+ text_w_pairs[effect_end:]
        text_w_pairs = text_w_pairs[:cause_start] + '<ARG0>' + text_w_pairs[cause_start:cause_end] + '</ARG0>'+ text_w_pairs[cause_end:]
        
    if ('<ARG0>' not in text_w_pairs) or ('<ARG1>' not in text_w_pairs):
        raise ValueError(f'Cause/Effect Arg not found: \n{text_w_pairs}')
    
    return text_w_pairs