import stanza
import pandas as pd
from stanza.utils.conll import CoNLL
from collections import defaultdict


# stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

def get_dep_pos(sentences):

    output = []
    for sent in sentences:
        doc = nlp(sent)
        output.append('\n'.join([f'id: {word.id}\tword: {word.text}\tpos: {word.xpos}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' 
            for sent in doc.sentences for word in sent.words]))
    
    return output


open_sesame_columns = ['ID',
 'FORM',
 'LEMMA',
 'PLEMMA',
 'POS',
 'PPOS',
 'SENT#',
 'PFEAT',
 'HEAD',
 'PHEAD',
 'DEPREL',
 'PDEPREL',
 'LU',
 'FRAME',
 'ROLE']


def sentence_to_conll(sent, sent_num=0):
    
    doc = nlp(sent) # doc is class Document
    dicts = doc.to_dict() # dicts is List[List[Dict]], representing each token / word in each sentence in the document
    conll = CoNLL.convert_dict(dicts) # conll is List[List[List]], representing each token / word in each sentence in the document

    df = []
    for d in CoNLL.convert_conll(conll)[0]:
        df.append([
            str(int(d['id'][0])),            # ID
            str(d['text'].encode('utf-8')),  # FORM
            "_", str(d['lemma']),            # LEMMA, PLEMMA
            "_", str(d['xpos']),             # POS, PPOS
            str(int(sent_num)), "_",         # SENT#, PFEAT
            "_", str(int(d['head']) + 1),    # HEAD, PHEAD
            "_", "_",                        # DEPREL, PDEPREL
            "_", "_", "O"                    # LU, FRAME, ROLE
        ])
    
    return df


def format_sents_for_opensesame(sentences):
    list_of_rows_df = []
    for sent_num, sent in enumerate(sentences):
        list_of_rows_df.extend(sentence_to_conll(sent, sent_num))

    # save file to data/apply
    # tally with globalconfig.py address
    conll_file = open("/home/fiona/OpenSESAME/data/apply/output.conll", "w")
    for row_num, row in enumerate(list_of_rows_df):
        if int(row[0])==1 and row_num!=0:
            conll_file.write("\n")
        conll_file.write('\t'.join(row) + "\n")
    conll_file.write("\n")
    conll_file.close()


def get_frames(sentences):

    list_of_rows_df = []
    for sent_num, sent in enumerate(sentences):
        list_of_rows_df.extend(sentence_to_conll(sent, sent_num))
    tmp_df = pd.DataFrame(list_of_rows_df, columns=open_sesame_columns)
    tmp_df['ID'] = tmp_df['ID'].astype(int)
    tmp_df['SENT#'] = tmp_df['SENT#'].astype(int)

    frames_df = pd.read_csv("/home/fiona/OpenSESAME/data/apply/output.conll", sep='\t', header=None)
    frames_df.columns = open_sesame_columns
    frames_df = frames_df.drop(columns=['FORM'])
    frames_df = frames_df.merge(tmp_df[['SENT#','ID','FORM']], how='left', on=['SENT#','ID'])
    frames_df = frames_df[open_sesame_columns]
    
    # Format
    output_dict = defaultdict(list)
    output = ''
    for i,row in frames_df.iterrows():

        if int(row.ID)==1 and i>0:
            sent_id = frames_df.loc[i-1]['SENT#']
            output_dict[sent_id].append(output)
            output='' # reset

        if row.FRAME!='_':
            output+=f'[{row.FRAME}]'
        
        form = str(row.FORM)[2:-1]
        prev_bio = 'O'
        prev_tag = None
        if row.ROLE=='O':
            if prev_bio == 'I':
                output = output[:-1]+f'</{prev_tag}> {form} '
            else:
                output += f'{form} '
            prev_bio = 'O'
            prev_tag = None
        else:
            bio, tag = row.ROLE.split('-')
            if bio == 'B':
                output += f'<{row.ROLE}>{form} '
            elif bio == 'S':
                output += f'<{row.ROLE}>{form}</{row.ROLE}> '
            else:
                output += f'{form} '
            prev_bio = bio
            prev_tag = tag
    sent_id = frames_df.loc[i-1]['SENT#']
    output_dict[sent_id].append(output)

    return output_dict
