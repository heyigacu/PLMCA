import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PDBbind21_Pocket_Dataset, collate_fn_PDBbind21_pocket
from sklearn.metrics import average_precision_score, matthews_corrcoef
from model import *
from model_ablation import *
from utils import *
from sklearn.cluster import DBSCAN

class PLCA_Pocket(nn.Module):
    def __init__(self, in_dim_dict={'prot_ankh':1536, 'prot_esm2':320, 'prot_phychem':42, 'prot_node_local':147, 'prot_edge_local':73, 'ligand':40}, 
                 max_protein_length=1500, max_ligand_length=225, dropout=0.1, hidden_dim=256, num_layers=2, ablation_name='raw'):
        super().__init__()
        self.num_layers = num_layers
        self.lig_mlp = MLP(in_dim_dict['ligand'], hidden_dim, hidden_dim)
        self.prot_llm1_mlp = MLP(in_dim_dict['prot_ankh'], hidden_dim, hidden_dim)
        self.prot_llm2_mlp = MLP(in_dim_dict['prot_esm2'], hidden_dim, hidden_dim)
        self.prot_phychem_mlp = MLP(in_dim_dict['prot_phychem'], hidden_dim, hidden_dim)
        self.prot_local_mlp = MLP(in_dim_dict['prot_node_local'], hidden_dim, hidden_dim)
        self.prot_node_mlp = MLP(hidden_dim*4, hidden_dim*2, hidden_dim)
        self.prot_edge_mlp = MLP(in_dim_dict['prot_edge_local'], hidden_dim, hidden_dim)
        self.lig_graph_encoders = nn.ModuleList([Cov2MolGraphTransformer(hidden_dim,num_heads=4) for _ in range(num_layers)]) 
        self.lig_pos_encoder = PositionalEncoding(hidden_dim, max_len=max_ligand_length)
        self.prot_graph_encoders=nn.ModuleList([Cov2ProtGraphTransformer(hidden_dim,num_heads=4) for _ in range(num_layers)])
        self.prot2lig_CA=nn.ModuleList([TransformerBlock(d_model=hidden_dim,num_heads=4,dropout=dropout) for _ in range(num_layers)])
        self.lig2prot_CA=nn.ModuleList([TransformerBlock(d_model=hidden_dim,num_heads=4,dropout=dropout) for _ in range(num_layers)])
        self.pocket_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.max_protein_length = max_protein_length
        self.max_ligand_length = max_ligand_length
        self.return_feature = True

    def forward(self, prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig):
        lig_g_node = self.lig_mlp(lig_graph.ndata['atom_feat'])
        prot_llm1 = self.prot_llm1_mlp(prot_graph.ndata['ankh'])        
        prot_llm2 = self.prot_llm2_mlp(prot_graph.ndata['esm2'])
        if self.return_feature:
            init_prot_node = pad_from_mask(prot_graph.ndata['esm2'], mask_prot) 
        prot_phychem = self.prot_phychem_mlp(prot_graph.ndata['phychem'])
        prot_node_local = self.prot_local_mlp(prot_graph.ndata['local'])
        prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2,prot_phychem,prot_node_local], dim=-1))
        prot_g_edge = self.prot_edge_mlp(prot_graph.edata['local'])
        for i in range(self.num_layers):
            lig_g_node = self.lig_graph_encoders[i](lig_graph, lig_g_node)
            lig_node = pad_from_mask(lig_g_node, mask_lig)
            if i ==0: 
                lig_node = self.lig_pos_encoder(lig_node) 
            prot_g_node, prot_g_edge = self.prot_graph_encoders[i](prot_graph, prot_g_node, prot_g_edge)  
            prot_node = pad_from_mask(prot_g_node, mask_prot) 
            prot_node = self.prot2lig_CA[i](q=prot_node,k=lig_node,v=lig_node,k_mask=mask_lig)*mask_prot.unsqueeze(-1).float()
            lig_node = self.lig2prot_CA[i](q=lig_node,k=prot_node,v=prot_node,k_mask=mask_prot)*mask_lig.unsqueeze(-1).float()
            prot_g_node = unpad_from_mask(prot_node, mask_prot)
            lig_g_node = unpad_from_mask(lig_node, mask_lig)
        pocket_logits = self.pocket_predictor(prot_node)
        if self.return_feature:
            return pocket_logits, init_prot_node, prot_node
        else:
            return pocket_logits

def train(split_method = 'unseen_protein', ablation_name='raw'):
    batch_size = 12
    hidden_dim = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ablation_name == 'raw':
        model = PLCA_Pocket(hidden_dim=hidden_dim).to(device)
    else:
        model = PLCA_Pocket_Ablation(hidden_dim=hidden_dim, ablation_name=ablation_name).to(device)
    train_dataset = PDBbind21_Pocket_Dataset(f'dataset/processed/PDBbind21_pocket/{split_method}/train.csv')
    val_dataset = PDBbind21_Pocket_Dataset(f'dataset/processed/PDBbind21_pocket/{split_method}/val.csv')
    test_dataset = PDBbind21_Pocket_Dataset(f'dataset/processed/PDBbind21_pocket/{split_method}/test.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_PDBbind21_pocket)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_PDBbind21_pocket)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_PDBbind21_pocket)
    optimizer = get_std_opt(len(train_dataset), batch_size=batch_size, parameters=model.parameters(), d_model=hidden_dim, top_lr=5e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    criterion_pocket = nn.BCEWithLogitsLoss(reduction='none')
    best_val_aupr = 0.
    patience, wait = 10, 0  
    result_dir = f"pretrained/pocket/{split_method}/{ablation_name}"
    os.makedirs(result_dir, exist_ok=True)
    best_model_path = f"{result_dir}/model.pt"

    for epoch in range(200):
        model.train()
        train_loss = 0
        num_batch = len(train_loader)
        for prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig in train_loader:
            prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig = \
                prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device)
            pocket_logits = model(prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig)
            loss_all=criterion_pocket(pocket_logits.squeeze(-1),pocket_label_padded)
            loss=(loss_all*mask_prot).sum()/mask_prot.sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/num_batch
        model.eval()
        val_loss = 0
        val_preds, val_probs, val_labels, val_masks = [], [], [], []
        num_batch = len(val_loader)
        with torch.no_grad():
            for prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig in val_loader:
                prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig = \
                    prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device)
                pocket_logits = model(prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig)
                loss_all=criterion_pocket(pocket_logits.squeeze(-1),pocket_label_padded)
                loss=(loss_all*mask_prot).sum()/mask_prot.sum()
                val_loss += loss.item()
                pocket_prob = torch.sigmoid(pocket_logits)
                pocket_pred = (pocket_prob > 0.5).float()
                val_preds.append(pocket_pred.squeeze(-1).cpu())
                val_probs.append((pocket_prob.squeeze(-1).cpu())*(mask_prot.cpu()))
                val_labels.append(pocket_label_padded.cpu())
                val_masks.append(mask_prot.cpu())
        val_loss = val_loss/num_batch
        val_preds=torch.cat(val_preds,0).numpy()
        val_probs=torch.cat(val_probs,0).numpy()
        val_labels=torch.cat(val_labels,0).numpy()
        val_masks=torch.cat(val_masks,0).numpy()

        valid_idx=val_masks.flatten()>0
        val_pocket_acc=(val_preds.flatten()[valid_idx]==val_labels.flatten()[valid_idx]).mean()
        idx=val_masks>0.5
        label=val_labels[idx]
        prob=val_probs[idx]
        pos_score = prob[label == 1]
        pos_mean = pos_score.mean()
        val_pocket_aupr=average_precision_score(val_labels.flatten()[valid_idx],val_probs.flatten()[valid_idx])
        val_pocket_mcc=matthews_corrcoef(val_labels.flatten()[valid_idx],val_preds.flatten()[valid_idx])
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}| Val Loss: {val_loss:.4f} {pos_mean} | Acc: {val_pocket_acc:.4f} | AUPR: {val_pocket_aupr:.4f} | MCC: {val_pocket_mcc:.4f}")
        if val_pocket_aupr > best_val_aupr:
            best_val_aupr = val_pocket_aupr
            torch.save(model.state_dict(), best_model_path)
            np.savetxt(f"{result_dir}/val_prob.txt", val_probs, fmt="%.4f")
            np.savetxt(f"{result_dir}/val_label.txt", val_labels, fmt="%.0f")
            np.savetxt(f"{result_dir}/val_mask.txt", val_masks, fmt="%.0f")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("⛔ Early stopping triggered.")
                break
    model.load_state_dict(torch.load(best_model_path))
    print(f"Training finished. Best Val AUPR: {best_val_aupr:.4f}")
    test_probs, test_labels, test_masks = [], [], []
    with torch.no_grad():
        for prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig in test_loader:
            prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig = \
                prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device)
            pocket_logits = model(prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig)
            pocket_prob = torch.sigmoid(pocket_logits)
            test_probs.append((pocket_prob.squeeze(-1).cpu())*(mask_prot.cpu()))
            test_labels.append(pocket_label_padded.cpu())
            test_masks.append(mask_prot.cpu())
        test_probs=torch.cat(test_probs,0).numpy()
        test_labels=torch.cat(test_labels,0).numpy()
        test_masks=torch.cat(test_masks,0).numpy()
        np.savetxt(f"{result_dir}/test_prob.txt", test_probs, fmt="%.4f")
        np.savetxt(f"{result_dir}/test_label.txt", test_labels, fmt="%.0f")
        np.savetxt(f"{result_dir}/test_mask.txt", test_masks, fmt="%.0f")


def coords_cluster(coords,predict_prob,threshold=0.5,eps=8.0,min_samples=3):
    assert coords.shape[0]==predict_prob.shape[0],"coords 与 predict_prob 长度不匹配"
    mask=predict_prob>threshold
    if mask.sum()==0:return [],np.full(len(coords),-1),None,None,None
    high_coords,high_probs=coords[mask],predict_prob[mask]
    db=DBSCAN(eps=eps,min_samples=min_samples)
    labels_sub=db.fit_predict(high_coords)
    labels=np.full(len(coords),-1);labels[mask]=labels_sub
    pockets=[]
    for k in np.unique(labels_sub):
        if k==-1:continue
        sel=labels_sub==k
        center=high_coords[sel].mean(0);score=high_probs[sel].mean()
        pockets.append({'id':int(k),'center':center,'size':int(sel.sum()),'score':float(score)})
    pockets.sort(key=lambda x:x['score'],reverse=True)
    if pockets:
        largest=max(pockets,key=lambda x:x['size'])
        center_x,center_y,center_z=largest['center']
    else:
        center_x=center_y=center_z=None
    return pockets,labels,center_x,center_y,center_z


from Bio.PDB import PDBParser
import numpy as np

def pdb_to_ca_coords(pdbpath):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdbpath)
    ca_coords = []
    chains = []
    resids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:  
                    atom = residue["CA"]
                    ca_coords.append(atom.get_coord())
                    chains.append(chain.id)
                    resids.append(residue.id[1]) 

    return np.array(ca_coords), chains, resids


def predict(task_name, cluster=True):
    task_dir = f"task/{task_name}/"
    df_path = f"{task_dir}/{task_name}.csv"
    df = pd.read_csv(df_path, sep='\t')
    best_model_path = f"pretrained/pocket/random/raw/model.pt"
    protein_graph_dir=f'{task_dir}/protein/mm_graphs'
    ligand_graph_dir=f'{task_dir}/ligand/graphs'
    esm2_dir=f'{task_dir}/protein/esm2'
    device = torch.device('cpu')
    batch_size = 12
    model = PLCA_Pocket().to(device)
    model.load_state_dict(torch.load(best_model_path))
    predict_dataset = PDBbind21_Pocket_Dataset(df_path, protein_graph_dir, ligand_graph_dir, esm2_dir, predict=True)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_PDBbind21_pocket)
    predict_probs, predict_masks = [], []
    model.eval()
    with torch.no_grad():
        for prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig in predict_loader:
            prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig = \
                prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device)
            pocket_logits = model(prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig)
            pocket_prob = torch.sigmoid(pocket_logits)
            predict_probs.append((pocket_prob.squeeze(-1).cpu())*(mask_prot.cpu()))
            predict_masks.append(mask_prot.cpu())
    predict_probs=torch.cat(predict_probs,0).numpy()
    predict_masks=torch.cat(predict_masks,0).numpy()
    pockets = []
    for i in range(len(df)):
        valid_probs=np.round(predict_probs[i][predict_masks[i]==1],3)
        pockets.append(' '.join([str(i) for i in valid_probs]))
    df['Pocket'] = pockets
    df.to_csv(f"{task_dir}/{task_name}_POCKET.csv", index=False, sep='\t')
    centers = []
    sizes = []

    if cluster:
        for index,row in df.iterrows():
            uniprot_id = row['Protein_UniProtID']
            # idx = np.where(valid_probs > 0.5)[0]
            top_k = 10
            idx = np.argsort(valid_probs)[-top_k:][::-1]
            print(idx+1)
            ca_coords,_,_ = pdb_to_ca_coords(f'{task_dir}/protein/pdbs/{uniprot_id}.pdb')  
            pocket_cas = ca_coords[np.array(idx)]
            db=DBSCAN(eps=10.0,min_samples=5)
            labels_sub=db.fit_predict(pocket_cas)
            valid_labels = labels_sub[labels_sub != -1]
            unique, counts = np.unique(valid_labels, return_counts=True)
            largest_label = unique[np.argmax(counts)]
            largest_cluster_idx = np.where(labels_sub == largest_label)[0]
            
            coords = pocket_cas[largest_cluster_idx]
            center = coords.mean(axis=0)        # (center_x, center_y, center_z)
            min_xyz = coords.min(axis=0)
            max_xyz = coords.max(axis=0)
            size = max_xyz - min_xyz + 10          # (size_x, size_y, size_z)
            center_x, center_y, center_z = center
            size_x, size_y, size_z = size + 10
            centers.append(center)
            sizes.append(size)
    df[['center_x','center_y','center_z']] = pd.DataFrame(centers, index=df.index)
    df[['size_x','size_y','size_z']]     = pd.DataFrame(sizes, index=df.index)
    df.to_csv(f"{task_dir}/{task_name}_POCKET.csv", index=False, sep='\t')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot():
    task_dir = f"dataset/public/"
    df_path = f"dataset/processed/PDBbind21_pocket/random/test.csv"
    df = pd.read_csv(df_path, sep='\t')
    best_model_path = f"pretrained/pocket/random/raw/model.pt"
    protein_graph_dir=f'{task_dir}/protein/mm_graphs'
    ligand_graph_dir=f'{task_dir}/ligand/graphs'
    esm2_dir=f'{task_dir}/protein/esm2'
    device = torch.device('cpu')
    batch_size = 12
    model = PLCA_Pocket().to(device)
    model.load_state_dict(torch.load(best_model_path))
    predict_dataset = PDBbind21_Pocket_Dataset(df_path, protein_graph_dir, ligand_graph_dir, esm2_dir, predict=False)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_PDBbind21_pocket)
    init_features, final_features = [], []
    pocket_labels_padded, mask_prots  = [], []
    model.eval()
    with torch.no_grad():
        for prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig in predict_loader:
            prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig = \
                prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device)
            pocket_logits, init_feature, final_feature  = model(prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig)
            init_feature = init_feature.reshape(-1, init_feature.size(-1))
            final_feature = final_feature.reshape(-1, final_feature.size(-1))
            pocket_label_padded = pocket_label_padded.reshape(-1)
            mask_prot = mask_prot.reshape(-1)
            init_features.append(init_feature.cpu())
            final_features.append(final_feature.cpu())
            pocket_labels_padded.append(pocket_label_padded.cpu())
            mask_prots.append(mask_prot.cpu())
    init_flat=torch.cat(init_features,0).numpy()
    final_flat=torch.cat(final_features,0).numpy()
    labels_flat=torch.cat(pocket_labels_padded,0).numpy()
    masks_flat=torch.cat(mask_prots,0).float().numpy()
    print(labels_flat.shape, masks_flat.shape)
    valid_idx = (masks_flat == 1)

    init_valid  = init_flat[valid_idx]
    final_valid = final_flat[valid_idx]
    labels_valid = labels_flat[valid_idx]

    pos_idx = (labels_valid == 1)
    neg_idx = (labels_valid == 0)

    # ---- t-SNE 1：初始特征 ----
    tsne1 = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
    init_tsne = tsne1.fit_transform(init_valid)

    # ---- t-SNE 2：最终特征 ----
    tsne2 = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
    final_tsne = tsne2.fit_transform(final_valid)

    # ---- 图1：初始特征 ----
    plt.figure(figsize=(6,5))
    plt.scatter(init_tsne[neg_idx,0], init_tsne[neg_idx,1], s=5, c="lightgray")
    plt.scatter(init_tsne[pos_idx,0], init_tsne[pos_idx,1], s=8, c="red")
    plt.title("t-SNE of Initial Features")
    plt.tight_layout()
    plt.savefig("tsne_init.png", dpi=300)

    # ---- 图2：最终特征 ----
    plt.figure(figsize=(6,5))
    plt.scatter(final_tsne[neg_idx,0], final_tsne[neg_idx,1], s=5, c="lightgray")
    plt.scatter(final_tsne[pos_idx,0], final_tsne[pos_idx,1], s=8, c="red")
    plt.title("t-SNE of Final Features")
    plt.tight_layout()
    plt.savefig("tsne_final.png", dpi=300)

    print("Saved tsne_init.png and tsne_final.png")


if __name__ == "__main__":
    # for split_method in ['random', 'unseen_protein', 'unseen_ligand', 'both_unseen']:
    #     train(split_method = split_method, ablation_name='raw')

    # split_method = 'unseen_protein'
    # for ablation in [ 'wo_ankh', 'wo_phychem', 'wo_geometric', 'wo_esm2']:
    #     train(split_method = split_method, ablation_name=ablation)

    # predict('5SQQ')
    plot()