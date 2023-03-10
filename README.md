# Steps

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
python process_extraction
```
Original workings at:
http://localhost:8888/notebooks/79%20Mass/notebooks/Process%20Extraction.ipynb

#### 4. Cluster Nodes
Depends on method

#### 5. Display Graph
