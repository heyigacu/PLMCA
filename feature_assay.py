import os,torch,numpy as np,pandas as pd
from transformers import AutoTokenizer,AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_sentence_length(df_path='dataset/ChEMBL/ChEMBL.csv',text_column='Assay Description',
                            tokenizer_name='tools/biobert',save_hist='result/statisitcs/assay_distribution.png'):
    df=pd.read_csv(df_path,sep='\t')
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
    texts=df[text_column].fillna('').astype(str).tolist()
    texts = list(set(texts))
    lengths=[len(tokenizer.tokenize(t)) for t in texts]
    s=pd.Series(lengths)
    print(f'mean:{s.mean():.2f} max:{s.max()}')
    plt.figure(figsize=(8,4))
    plt.hist(lengths,bins=50,color='steelblue',edgecolor='black')
    plt.xlabel('Tokens');plt.ylabel('Number of Setences');plt.title(f'Distribution of Setence Length')
    if save_hist:plt.savefig(save_hist,dpi=300,bbox_inches='tight')
 
def generate_all_assay_biobert(df_path='dataset/ChEMBL/ChEMBL.csv',save_folder='dataset/feature/assay/biobert/',assay_column='Assay_ChEMBL_ID'):
    tokenizer = AutoTokenizer.from_pretrained("tools/biobert")
    model = AutoModel.from_pretrained("tools/biobert")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    os.makedirs(save_folder,exist_ok=True)
    df=pd.read_csv(df_path,sep='\t')
    error_ls=[]
    def get_embed(text,max_len=100):
        inputs=tokenizer(text,return_tensors='pt',truncation=True,max_length=max_len)
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():out=model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
        return out
    for _,row in df.iterrows():
        assay_id=str(row[assay_column])
        save_path=os.path.join(save_folder,f'{assay_id}.npz')
        if os.path.exists(save_path):continue
        try:
            desc=str(row.get('Assay Description',''))
            if not desc or desc=='nan':continue
            np.savez_compressed(save_path,biobert=get_embed(desc))
        except Exception as e:
            error_ls.append(assay_id)
    print(f'Failed embeddings: {len(error_ls)} →',error_ls)

def load_all_assay_biobert(df,save_folder='dataset/feature/assay/biobert/',assay_column='Assay_ChEMBL_ID'):
    assay_ids=set(df[assay_column].astype(str))
    feat_dict={}
    for aid in assay_ids:
        path=os.path.join(save_folder,f'{aid}.npz')
        if not os.path.exists(path):continue
        feat=np.load(path)['biobert'].astype(np.float32)
        feat_dict[aid]=torch.tensor(feat,dtype=torch.float32) 
    return feat_dict


if __name__ == '__main__':
    # analyze_sentence_length()
    generate_all_assay_biobert()