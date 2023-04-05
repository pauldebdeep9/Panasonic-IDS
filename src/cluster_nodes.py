import os, re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from simcse import SimCSE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import pickle


def _preprocess_text(documents):
    """ Basic preprocessing of text
    Steps:
        * Lower text
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
    """
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]

    return cleaned_documents


class TFIDF_IDFi(TfidfTransformer):

    def __init__(self, X_per_cluster, X_origin, all_documents, *args, **kwargs):
        print('====== Using TFIDF_IDFi ======')
        super().__init__(*args, **kwargs)
        self.X_per_cluster = X_per_cluster
        self.X_origin = X_origin
        self.all_documents = all_documents
        
    
    def score(self):
        
        self._global_tfidf = self.fit_transform(self.X_origin)
        
        global_df = pd.DataFrame(self._global_tfidf.toarray())
        global_df['Topic'] = self.all_documents.Topic
        
        avg_global_df = global_df.groupby(['Topic'], as_index=False).mean()
        avg_global_df = avg_global_df.drop('Topic', 1)
        self._avg_global_tfidf = avg_global_df.values
        
        local_tfidf_transformer = TfidfTransformer()
        local_tfidf_transformer.fit_transform(self.X_per_cluster)
        self._idfi = local_tfidf_transformer.idf_
        
        scores = self._avg_global_tfidf * self._idfi
        scores = normalize(scores, axis=1, norm='l1', copy=False)
        scores = sp.csr_matrix(scores)

        return scores 
    

def _top_n_idx_sparse(matrix, n):
    """ Return indices of top n values in each row of a sparse matrix
    Retrieved from:
        https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix
    Args:
        matrix: The sparse matrix from which to get the top n indices per row
        n: The number of highest values to extract from each row
    Returns:
        indices: The top n indices per row
    """
    indices = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)
    return np.array(indices)


def _top_n_values_sparse(matrix, indices):
    """ Return the top n values for each row in a sparse matrix
    Args:
        matrix: The sparse matrix from which to get the top n indices per row
        indices: The top n indices per row
    Returns:
        top_values: The top n scores per row
    """
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
        top_values.append(scores)
    return np.array(top_values)


def cluster_nodes(graph_folder):
    csv_file_path = os.path.join(graph_folder,f"nodes_extracted_infos.pkl")
    data = pd.read_pickle(csv_file_path)
    print(f'working on: {csv_file_path}')
    print(f"length of data: {len(data)}")
    
    N_TOPICS = 3000 #int(len(data)*2*0.1)
    TOP_N_WORDS = 15
    TOP_N_KEYWORDS = 5
    LOAD_EMBEDDINGS = True

    sentences = list(data['cause_action_rem'])+list(data['effect_action_rem'])
    sentences = [re.sub('[MASK]','',i) for i in sentences]
    emb_file_path = os.path.join(graph_folder,f"embeddings.pkl")
    if LOAD_EMBEDDINGS:
        filehandler = open(emb_file_path,"rb")
        embeddings = pickle.load(filehandler)
        filehandler.close()
        print(f"embeddings loaded from: {emb_file_path}")
    else:
        # create word embeddings
        print(f'generating embeddings...')
        model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        embeddings = model.encode(sentences)
        print(f"embeddings.shape: {embeddings.shape}")
        filehandler = open(emb_file_path,"wb")
        pickle.dump(embeddings,filehandler)
        filehandler.close()
        print(f"embeddings saved to: {emb_file_path}")
    
    print(f'clustering...')
    print(f"N_TOPICS: {N_TOPICS}")
    # perform clustering
    kmeans = KMeans(N_TOPICS)
    kmeans_output = kmeans.fit(embeddings)
    # _update_topic_size
    documents = pd.DataFrame({
        'Topic': list(kmeans_output.labels_),
        'Document': sentences
    })
    sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
    topic_sizes = dict(zip(sizes.Topic, sizes.Document))
    # _extract_topics
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    # _weighting_words
    concatenated_documents = _preprocess_text(documents_per_topic.Document.values)
    origin_documents = _preprocess_text(documents.Document.values)
    # count the words in a cluster
    vectorizer_model = CountVectorizer()
    vectorizer_model.fit(concatenated_documents)
    words = vectorizer_model.get_feature_names()
    # k * vocab
    X_per_cluster = vectorizer_model.transform(concatenated_documents)
    # D * vocab
    X_origin = vectorizer_model.transform(origin_documents)

    # Get topics
    scores = TFIDF_IDFi(X_per_cluster, X_origin, documents).score()
    # _extract_words_per_topic
    labels = sorted(list(topic_sizes.keys()))
    indices = _top_n_idx_sparse(scores, 30)
    scores = _top_n_values_sparse(scores, indices)
    sorted_indices = np.argsort(scores, 1)
    indices = np.take_along_axis(indices, sorted_indices, axis=1)
    scores = np.take_along_axis(scores, sorted_indices, axis=1)
    topics = {label: [(words[word_index], score)
                    if word_index and score > 0
                    else ("", 0.00001)
                    for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                    ]
            for index, label in enumerate(labels)}
    topics = {label: values[:TOP_N_WORDS] for label, values in topics.items()}

    # Document topic2keywords
    topic2keywords = {}
    store_to_logger = ''
    for k,v in topics.items():
        store_to_logger+=f'\n\n{k}\n{v}'
        if topic_sizes[k]==1:
            topic2keywords[k] = documents[documents['Topic']==k]['Document'].iloc[0]
        else:
            topic2keywords[k] = '_'.join([i[0] for i in v[:TOP_N_KEYWORDS]])
    
    # Save files
    documents['Keywords'] = documents['Topic'].apply(lambda x: topic2keywords[int(x)])
    documents.to_csv(
        os.path.join(graph_folder,"spans_topics_keywords.csv"), 
        index=False, encoding='utf-8-sig'
        )

    with open(os.path.join(graph_folder,"top_keywords_per_topic.txt"), 'w') as f:
        f.write(store_to_logger)
    
    with open(os.path.join(graph_folder,"topic2keywords.json"), "w") as fp:
        json.dump(topic2keywords,fp) 

    filehandler = open(os.path.join(graph_folder,"kmeans.pkl"),"wb")
    pickle.dump(kmeans_output,filehandler)
    filehandler.close()

    # Append topic cols
    data['cause_topic'] = list(documents['Topic'])[:int(len(documents)/2)]
    data['effect_topic'] = list(documents['Topic'])[int(len(documents)/2):]

    # Reparse graph infos
    headers = ['cause','effect','source','support','evidence']
    rows = []
    for i,row in data.iterrows():
        rows.append([
            str(int(row.cause_topic))+'>>'+topic2keywords[int(row.cause_topic)],
            str(int(row.effect_topic))+'>>'+topic2keywords[int(row.effect_topic)],
            str(row.source),
            int(row['support']),
            str(row.cause_action_rem)+' --> '+ str(row.effect_action_rem) + ';' + str(row.evidence)
        ])
    graph_df = pd.DataFrame(rows, columns=headers)
    # get nodes
    node_df = pd.concat([
        graph_df[['cause','source']].rename(columns={'cause':'node','source':'sources'}),
        graph_df[['effect','source']].rename(columns={'effect':'node','source':'sources'})
    ],axis=0).drop_duplicates().reset_index(drop=True)
    node_df = node_df.groupby(['node'])['sources'].apply(lambda x: ', '.join(x)).reset_index()

    # processed graph (after clustering)
    graph_df.to_csv(os.path.join(graph_folder,'graph_clustered.csv'), index=False, encoding='utf-8-sig')
    node_df.to_csv(os.path.join(graph_folder,'node_df_clustered.csv'), index=False, encoding='utf-8-sig')


if __name__ == "__main__":   
    graph_folder = r"D:\66 CausalMap\SciLit_CausalMap\visualization\mir_paper"
    cluster_nodes(graph_folder)