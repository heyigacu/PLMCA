import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PDBbind21_Affinity_Dataset, collate_fn_PDBbind21_affinity
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import average_precision_score, matthews_corrcoef
from model import *
from model_ablation import *
from utils import *
from sklearn.cluster import DBSCAN



class PLCA_Affinity(nn.Module):
    def __init__(self, in_dim_dict={'prot_ankh':1536, 'prot_esm2':320, 'prot_phychem':42, 'prot_node_local':147, 'prot_edge_local':73, 'ligand':40}, 
                 max_protein_length=1500, max_ligand_length=225, dropout=0.1, hidden_dim=256, num_layers=2, ablation_name='raw', num_pred_heads=1, use_assay=False):
        super().__init__()
        self.num_layers = num_layers
        self.lig_mlp = MLP(in_dim_dict['ligand'], hidden_dim, hidden_dim)
        self.prot_llm1_mlp = MLP(in_dim_dict['prot_ankh'], hidden_dim, hidden_dim)
        self.prot_llm2_mlp = MLP(in_dim_dict['prot_esm2'], hidden_dim, hidden_dim)
        self.prot_phychem_mlp = MLP(in_dim_dict['prot_phychem'], hidden_dim, hidden_dim)
        self.prot_local_mlp = MLP(in_dim_dict['prot_node_local'], hidden_dim, hidden_dim)
        self.prot_node_mlp = MLP(hidden_dim*4, hidden_dim*2, hidden_dim)
        self.prot_edge_mlp = MLP(in_dim_dict['prot_edge_local'], hidden_dim, hidden_dim)
        self.lig_graph_encoders = nn.ModuleList([Cov2MolMPNN(hidden_dim,num_heads=4) for _ in range(num_layers)]) 
        self.lig_pos_encoder = PositionalEncoding(hidden_dim, max_len=max_ligand_length)
        self.prot_graph_encoders=nn.ModuleList([Cov2ProtMPNN(hidden_dim,num_heads=4) for _ in range(num_layers)])
        self.prot2lig_CA=nn.ModuleList([TransformerBlock(d_model=hidden_dim,num_heads=4,dropout=dropout) for _ in range(num_layers)])
        self.lig2prot_CA=nn.ModuleList([TransformerBlock(d_model=hidden_dim,num_heads=4,dropout=dropout) for _ in range(num_layers)])
        self.use_assay = use_assay
        if self.use_assay:
            self.assay_1d_encoders = nn.ModuleList([Seq1DCNN(in_dim_dict['assay'], hidden_dim, kernel_size=5, num_layers=2, dropout=dropout)]) 
            self.shared_fc = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        else:
            self.shared_fc = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.affinity_headers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim//2, hidden_dim//4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim//4, 1)
                ) for _ in range(num_pred_heads)
            ])
        self.max_protein_length = max_protein_length
        self.max_ligand_length = max_ligand_length

    def forward(self, prot_graph, esm2_padded, lig_graph, assay_feat, mask_prot, mask_lig, mask_assay, pocket_label_padded):
        lig_g_node = self.lig_mlp(lig_graph.ndata['atom_feat'])
        prot_llm1 = self.prot_llm1_mlp(prot_graph.ndata['ankh'])        
        prot_llm2 = self.prot_llm2_mlp(prot_graph.ndata['esm2'])
        prot_phychem = self.prot_phychem_mlp(prot_graph.ndata['phychem'])
        prot_node_local = self.prot_local_mlp(prot_graph.ndata['local'])
        prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2,prot_phychem,prot_node_local], dim=-1))
        del prot_llm1
        del prot_llm2
        del prot_phychem
        del prot_node_local
        prot_g_edge = self.prot_edge_mlp(prot_graph.edata['local'])
        for i in range(self.num_layers):
            feat_assay = self.assay_1d_encoders[i](assay_feat) * mask_assay.unsqueeze(-1).float() if self.use_assay else None # [batch,len,dim]
            lig_g_node = self.lig_graph_encoders[i](lig_graph, lig_g_node)
            lig_node = pad_from_mask(lig_g_node, mask_lig)
            if i ==0: 
                lig_node = self.lig_pos_encoder(lig_node) 
            prot_g_node, prot_g_edge = self.prot_graph_encoders[i](prot_graph, prot_g_node, prot_g_edge)  
            prot_node = pad_from_mask(prot_g_node, mask_prot) 
            prot_node = self.prot2lig_CA[i](q=prot_node,k=lig_node,v=lig_node,k_mask=mask_lig)*mask_prot.unsqueeze(-1).float()
            lig_node = self.lig2prot_CA[i](q=lig_node,k=prot_node,v=prot_node,k_mask=mask_prot)*mask_lig.unsqueeze(-1).float()
            prot_g_node = unpad_from_mask(prot_node, mask_prot)
            lig_g_node = unpad_from_mask(prot_node, mask_lig)
        del prot_g_node 
        del lig_g_node
        feat_prot = prot_node.mean(dim=1) # [batch,dim]
        feat_lig = lig_node.mean(dim=1)
        feat_assay = feat_assay.mean(dim=1) if self.use_assay else None
        feat_total = torch.cat([feat_prot, feat_lig, feat_assay], dim=-1) if self.use_assay else torch.cat([feat_prot, feat_lig], dim=-1) # [batch,dim*3] or #[batch,dim*2]                               
        feat_total = self.shared_fc(feat_total)
        affinitys = []
        for i, affinity_predictor in enumerate(self.affinity_headers):
            affinitys.append(affinity_predictor(feat_total))
        return torch.cat(affinitys, dim=1) # [batch,num_heads]


    
def train(dataset_name='PDBbind21_Kd', split_method = 'unseen_protein', ablation_name='raw', use_assay=False, 
          protein_column = 'Protein_UniProtID', ligand_column = 'Ligand_ID', num_layers=1, batch_size=64):
    batch_size = batch_size
    hidden_dim = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ablation_name == 'raw':
        model = PLCA_Affinity(hidden_dim=hidden_dim, use_assay=use_assay, num_layers=num_layers).to(device)
    elif ablation_name == 'layer=1':
        model = PLCA_Affinity(hidden_dim=hidden_dim, use_assay=use_assay, num_layers=1).to(device)
    else:
        model = PLCA_Affinity_Ablation(hidden_dim=hidden_dim, ablation_name=ablation_name).to(device)
    train_dataset = PDBbind21_Affinity_Dataset(f'dataset/processed/{dataset_name}/{split_method}/train.csv', phase='train', use_assay=use_assay,  protein_column=protein_column, ligand_column=ligand_column)
    val_dataset = PDBbind21_Affinity_Dataset(f'dataset/processed/{dataset_name}/{split_method}/val.csv', phase='val', use_assay=use_assay,  protein_column=protein_column, ligand_column=ligand_column)
    test_dataset = PDBbind21_Affinity_Dataset(f'dataset/processed/{dataset_name}/{split_method}/test.csv', phase='test', use_assay=use_assay,  protein_column=protein_column, ligand_column=ligand_column)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_PDBbind21_affinity)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_PDBbind21_affinity)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_PDBbind21_affinity)
    optimizer = get_std_opt(len(train_dataset), batch_size=batch_size, parameters=model.parameters(), d_model=hidden_dim, top_lr=5e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001)
    criterion_affinity = nn.MSELoss()
    best_val_mse = float('inf')
    patience, wait = 10, 0   
    result_dir = f"pretrained/affinity/{dataset_name}/{split_method}/{ablation_name}"
    os.makedirs(result_dir, exist_ok=True)
    best_model_path = f"{result_dir}/model.pt"

    for epoch in range(200):
        model.train()
        train_loss = 0
        num_batch = len(train_loader)
        for prot_graph, esm2_padded, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay in train_loader:
            prot_graph, esm2_padded, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay = \
                prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), assay_padded.to(device), affinity_label.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device), mask_assay.to(device)
            affinity_logits = model(prot_graph, esm2_padded, lig_graph, assay_padded, mask_prot, mask_lig, mask_assay, pocket_label_padded)
            loss=criterion_affinity(affinity_logits, affinity_label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/num_batch
        model.eval()
        val_loss = 0
        val_affinity_preds, val_affinity_labels = [], []
        num_batch = len(val_loader)
        with torch.no_grad():
            for prot_graph, esm2_padded, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay in val_loader:
                prot_graph, esm2_padded, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay = \
                    prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), assay_padded.to(device), affinity_label.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device), mask_assay.to(device)
                affinity_logits = model(prot_graph, esm2_padded, lig_graph, assay_padded, mask_prot, mask_lig, mask_assay, pocket_label_padded)
                loss=criterion_affinity(affinity_logits,affinity_label)
                val_loss += loss.item()
                val_affinity_preds.append(affinity_logits.cpu())
                val_affinity_labels.append(affinity_label.cpu())             
        val_affinity_labels = torch.cat(val_affinity_labels, dim=0).numpy()
        val_affinity_preds = torch.cat(val_affinity_preds, dim=0).numpy()
        val_affinity_mse=mean_squared_error(val_affinity_labels.flatten(),val_affinity_preds.flatten()) 
        val_affinity_r2=r2_score(val_affinity_labels.flatten(),val_affinity_preds.flatten())
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/num_batch:.4f} | MSE: {val_affinity_mse:.4f} | R²: {val_affinity_r2:.4f}")
        if val_affinity_mse < best_val_mse:
            best_val_mse = val_affinity_mse
            torch.save(model.state_dict(), best_model_path)
            np.savetxt(f"{result_dir}/val_prob.txt", val_affinity_preds, fmt="%.4f")
            np.savetxt(f"{result_dir}/val_label.txt", val_affinity_labels, fmt="%.4f")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
    model.load_state_dict(torch.load(best_model_path))
    print(f"Training finished. Best Val AUPR: {best_val_mse:.4f}")
    test_affinity_preds, test_affinity_labels = [], []
    with torch.no_grad():
        for prot_graph, esm2_padded, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay in test_loader:
            prot_graph, esm2_padded, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay = \
                prot_graph.to(device), esm2_padded.to(device), lig_graph.to(device), assay_padded.to(device), affinity_label.to(device), pocket_label_padded.to(device), mask_prot.to(device), mask_lig.to(device), mask_assay.to(device)
            affinity_logits = model(prot_graph, esm2_padded, lig_graph, assay_padded, mask_prot, mask_lig, mask_assay, pocket_label_padded)
            test_affinity_preds.append(affinity_logits.cpu())
            test_affinity_labels.append(affinity_label.cpu())          
    test_affinity_labels = torch.cat(test_affinity_labels, dim=0).numpy()
    test_affinity_preds = torch.cat(test_affinity_preds, dim=0).numpy()
    np.savetxt(f"{result_dir}/test_prob.txt", test_affinity_preds, fmt="%.4f")
    np.savetxt(f"{result_dir}/test_label.txt", test_affinity_labels, fmt="%.4f")





if __name__ == "__main__":
    split_method='unseen_protein'
    # for ablation in [ 'wo_ankh', 'wo_phychem', 'wo_geometric', 'wo_esm2']: # 'layer=1', 
    #     train(dataset_name='PDBbind21_Ki', split_method = split_method, ablation_name=ablation, ligand_column='Ligand_ID')
    # for ablation in ['wo_ankh', 'wo_phychem', 'wo_geometric', 'wo_esm2']: # 'layer=1',
    #     train(dataset_name='PDBbind21_Kd', split_method = split_method, ablation_name=ablation, ligand_column='Ligand_ID')
    # for split_method in ['random', 'unseen_protein', 'unseen_ligand']:

    # for split_method in [ 'unseen_protein']:
    #     train(dataset_name='PDBbind21_Kd', split_method = split_method, ablation_name='raw', ligand_column='Ligand_ID')
        # train(dataset_name='PDBbind21_Ki', split_method = split_method, ablation_name='raw', ligand_column='Ligand_ID')
    #     train(dataset_name='Davis', split_method = split_method, ablation_name='raw', ligand_column='Ligand_CID')
    # for split_method in ['unseen_ligand']:
    #     train(dataset_name='Davis', split_method = split_method, ablation_name='raw', ligand_column='Ligand_CID')
    for split_method in ['random', 'unseen_protein', 'unseen_ligand']:
        # train(dataset_name='KiBA', split_method = split_method, ablation_name='raw', ligand_column='Ligand_ChEMBL_ID')
        train(dataset_name='Davis', split_method = split_method, ablation_name='raw', ligand_column='Ligand_CID')
