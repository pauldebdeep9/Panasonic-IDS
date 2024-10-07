from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode the input
inputs = tokenizer.encode("The quick brown fox", return_tensors="pt")

# Generate text
outputs = model.generate(inputs, max_length=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))










# Steps

#### Set up Environment
conda env create -f cre_environment.yml

#### 1. Clean Data
http://localhost:8888/notebooks/66%20CausalMap/Panasonic-IDS/notebooks/EDA.ipynb
* De-duplication
* Fix sentence tokenization

#### 2. Parse/Extract Relations
Go to `/home/fiona/79 Mass`, run `mass_panasonic.py`

#### 3. Post-process
```
cd "D:\66 CausalMap\Panasonic-IDS\src"
conda activate cre
python process_extraction.py
```
Original workings at:
http://localhost:8888/notebooks/79%20Mass/notebooks/Process%20Extraction.ipynb

#### 4. Cluster Nodes
```
cd "D:\66 CausalMap\Panasonic-IDS\src"
conda activate cre
python convert_to_graph.py
python get_nodes_infos.py
python cluster_nodes.py
```

#### 5. Display Graph



#### Test Set
Get Annotations
1. http://localhost:8888/notebooks/66%20CausalMap/Panasonic-IDS/notebooks/Format%20Annotated%20Data.ipynb
2. Fix annotations in CSV
3. http://localhost:8888/notebooks/66%20CausalMap/Panasonic-IDS/notebooks/Format%20Annotated%20Data.ipynb
Get Predictions
4. http://localhost:8888/notebooks/66%20CausalMap/Panasonic-IDS/notebooks/Evaluate%20Extraction.ipynb
Evaluate Extraction, Store Results
5. http://localhost:8888/notebooks/66%20CausalMap/Panasonic-IDS/notebooks/EDA_nerevaluate.ipynb
6. "D:\66 CausalMap\Paper\paper_tables.xlsx"
