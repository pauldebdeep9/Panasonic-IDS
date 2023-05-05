import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'


def get_ce_cls(sentences):
    num_labels = 2
    cache_dir = None
    model_name_or_path = "tanfiona/unicausal-seq-baseline" #"bert-base-cased"

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True
    )

    batch_size = 128
    rounds = int(np.ceil(len(sentences)/batch_size))
    pred_clsses = []

    for r in range(1,rounds+1):
        
        input_text_tokenized = tokenizer(
            sentences[(r-1)*batch_size:r*batch_size],
            truncation=True, 
            padding=True,
            return_tensors="pt"
        )['input_ids'].to(device)

        prediction = model(input_text_tokenized) # Returns predicted probabilities per class
        pred_cls = torch.argmax(prediction.logits, dim=1) # Converts probabilities to sum to one
        pred_clsses.extend(pred_cls.detach().cpu().numpy().tolist())

    return pred_clsses


def get_text_w_pairs(tokens, ce_list):
    """
    tokens [list] : White-space separated list of word tokens.
    ce_list [list] : List of Cause-Effect tags. Either in ['C','E','O'] format, or BIO-format ['B-C','I-C'...]
    """
    
    # Sanity check
    assert(len(tokens)==len(ce_list))
    
    # Loop per token
    curr_ce = prev_ce = None
    for i, (tok, ce) in enumerate(zip(tokens, ce_list)):

        curr_ce = ce.split('-')[-1]

        if curr_ce!=prev_ce: # we only need to tag BOUNDARIES

            # opening
            if curr_ce=='C':
                tokens[i]='<ARG0>'+tok
            elif curr_ce=='E':
                tokens[i]='<ARG1>'+tok

            # closing
            if prev_ce=='C':
                tokens[i-1]=tokens[i-1]+'</ARG0>'
            elif prev_ce=='E':
                tokens[i-1]=tokens[i-1]+'</ARG1>'

        # update
        prev_ce = curr_ce

    # LAST closure
    if prev_ce=='C':
        tokens[i]=tokens[i]+'</ARG0>'
    elif prev_ce=='E':
        tokens[i]=tokens[i]+'</ARG1>'
        
    return ' '.join(tokens)


def get_ce_span(sentences):
    label_to_id = {'B-C': 0, 'B-E': 1, 'I-C': 2, 'I-E': 3, 'O': 4}
    num_labels = len(label_to_id)
    cache_dir = None
    model_name_or_path = "tanfiona/unicausal-tok-baseline" #"bert-base-cased"
    id_to_label = {}
    for k,v in label_to_id.items():
        id_to_label[v]=k

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True
    )

    outputs = []

    for sent in sentences:
        input_text_tokenized = tokenizer.encode(
            sent.split(' '),
            truncation=True, 
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        ).to(device)

        splitted_text = []
        for tok in input_text_tokenized[0]:
            splitted_text.append(tokenizer.decode(tok))
        
        predicted = model(input_text_tokenized) # Returns predicted probabilities per class
        predicted_probs = torch.nn.functional.softmax(predicted.logits, dim=1) # Converts probabilities to sum to one
        predicted_class = list(torch.argmax(predicted_probs, dim=2).detach().cpu().numpy()[0]) # Argmax would give us our predicted class
        predicted_labels = [id_to_label[i] for i in predicted_class]
        outputs.append(get_text_w_pairs(
            tokens=splitted_text[1:-1], 
            ce_list=predicted_labels[1:-1]
        ))
    
    return outputs