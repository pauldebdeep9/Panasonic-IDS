import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'


def get_ce_pair_model():
    num_labels = 2
    cache_dir = None
    model_name_or_path = "tanfiona/unicausal-pair-baseline"

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

    return model, tokenizer


def get_ce_pair_cls(model, tokenizer, sentences):
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