
import os
import torch
import dgl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from feature_protein import load_all_protein_multimodal_graph, load_all_protein_ankh, load_all_protein_esm2
from feature_ligand import load_all_ligand_graph,load_all_ligand_kpgt
from feature_assay import load_all_assay_biobert

# def statistics_of_PDBbind_dataset(csv_path='dataset/PDBbind/PDBbind21_SC_UR_UL_UP_pocket.csv'):
#     df = pd.read_csv(csv_path, sep='\t')
#     all_prot_dict, all_lig_dict = load_all_protein_esm2(df), load_all_ligand_graphs(df)
#     total_residues = total_pocket_residues = max_prot_length = max_lig_length = max_pocket_length =0

#     for _, row in df.iterrows():
#         prot_id, lig_id = row['Protein_UniProtID'], row['Ligand_ID']
#         protein_length, ligand_length = len(all_prot_dict[prot_id]), all_lig_dict[lig_id].num_nodes()
#         pocket_residues = [int(i) + int(row['PDBBind_UniProt_ResStart']) - int(row['PDBBind_PDB_ResStart']) - 1
#                            for i in row['PDBBind_Pocket_ResID'].split('_')]
#         total_residues += protein_length
#         total_pocket_residues += len(pocket_residues)
#         max_prot_length, max_lig_length, max_pocket_length = max(max_prot_length, protein_length), max(max_lig_length, ligand_length), max(max_pocket_length, len(pocket_residues))

#     ratio = total_pocket_residues / total_residues
#     pos_weight = (1 - ratio) / ratio 
#     return max_prot_length, max_lig_length, max_pocket_length, ratio , pos_weight

# def statistics_of_ChEMBL_dataset(csv_path='dataset/ChEMBL/ChEMBL.csv'):
#     df = pd.read_csv(csv_path, sep='\t')
#     all_prot_dict, all_lig_dict = load_all_protein_esm2(df), load_all_ligand_graphs(df, ligand_column='Ligand_ChEMBL_ID')
#     max_prot_length = max_lig_length = 0
#     prot_ids, lig_ids = [], []
#     for _, row in df.iterrows():
#         prot_id, lig_id = row['Protein_UniProtID'], row['Ligand_ChEMBL_ID']
#         prot_ids.append(prot_id)
#         lig_ids.append(lig_id)
#     # for prot_id in prot_ids:
#     #     protein_length = len(all_prot_dict[prot_id])
#     #     if max_prot_length < protein_length:
#     #         max_prot_length = protein_length
#     lig_ids= list(set(lig_ids))
#     print(len(lig_ids))
#     for lig_id in lig_ids:
#         ligand_length = all_lig_dict[lig_id].num_nodes()
#         if max_lig_length < ligand_length:
#             max_lig_length = ligand_length
#     return max_prot_length, max_lig_length

class PDBbind21_Pocket_Dataset(Dataset):
    def __init__(self, csv_path='dataset/dataset/PDBbind21_pocket/unseen_protein/train.csv', 
                 protein_graph_dir='dataset/public/protein/mm_graphs', 
                 ligand_graph_dir='dataset/public/ligand/graphs', 
                 esm2_dir='dataset/public/protein/esm2', 
                 max_lig_atoms=225,
                 max_prot_residues=1500, 
                 predict=False,
                 protein_column='Protein_UniProtID',
                 ligand_column='Ligand_ID'
                 ):
        self.df = pd.read_csv(csv_path, sep='\t', index_col=None if predict else 0).iloc[:24, :]
        self.lig_graph_dict = load_all_ligand_graph(self.df, ligand_graph_dir, ligand_column=ligand_column)
        self.prot_graph_dict = load_all_protein_multimodal_graph(self.df, protein_graph_dir)
        self.esm2_dict = load_all_protein_esm2(self.df, esm2_dir)
        self.max_lig_atoms = max_lig_atoms
        self.max_prot_residues = max_prot_residues
        self.predict = predict
        self.protein_column = protein_column
        self.ligand_column = ligand_column
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prot_graph = self.prot_graph_dict[row[self.protein_column]]
        protein_length = prot_graph.num_nodes()
        lig_graph = self.lig_graph_dict[row[self.ligand_column]]
        if not self.predict:
            pocket_label = torch.from_numpy(np.asarray(list(row['Label']),dtype=int))
            pocket_label_padded = torch.zeros(self.max_prot_residues, dtype=torch.float32)
            pocket_label_padded[:protein_length] = pocket_label[:self.max_prot_residues]
        else:
            pocket_label_padded = torch.zeros(self.max_prot_residues, dtype=torch.long)
        ligand_length = lig_graph.num_nodes()
        esm2_padded  = torch.tensor([0.0], dtype=torch.float32)
        # esm2_feat = self.esm2_dict[row[self.protein_column]]
        # esm2_padded = torch.zeros(self.max_prot_residues, esm2_feat.shape[1], dtype=torch.float32)
        # esm2_padded[:protein_length] = esm2_feat[:self.max_prot_residues]
        mask_prot = torch.zeros(self.max_prot_residues, dtype=torch.bool)
        mask_prot[:protein_length] = True
        mask_lig = torch.zeros(self.max_lig_atoms, dtype=torch.bool)
        mask_lig[:ligand_length] = True
        return prot_graph, esm2_padded, lig_graph, pocket_label_padded, mask_prot, mask_lig

def collate_fn_PDBbind21_pocket(batch):
    prot_graphs, esm2_feats, lig_graphs, pocket_labels_padded, mask_prots, mask_ligs = zip(*batch)
    lig_graphs = dgl.batch(lig_graphs)
    prot_graphs = dgl.batch(prot_graphs)
    esm2_feats = torch.stack(esm2_feats, dim=0)
    pocket_labels_padded = torch.stack(pocket_labels_padded, dim=0)
    mask_prots = torch.stack(mask_prots, dim=0)
    mask_ligs = torch.stack(mask_ligs, dim=0)
    return prot_graphs, esm2_feats, lig_graphs, pocket_labels_padded, mask_prots, mask_ligs



class PDBbind21_Affinity_Dataset(Dataset):
    def __init__(self, csv_path='dataset/dataset/PDBbind21_pocket/unseen_protein/train.csv', 
                 protein_graph_dir='dataset/public/protein/mm_graphs', 
                 ligand_graph_dir='dataset/public/ligand/graphs', 
                 kpgt_dir='dataset/public/ligand/kpgt', 
                 max_lig_atoms=225,
                 max_prot_residues=1500, 
                 phase='train',
                 protein_column='Protein_UniProtID',
                 ligand_column='Ligand_ID',
                 use_pocket=False,
                 use_assay=False
                 ):
        self.df = pd.read_csv(csv_path, sep='\t', index_col=0)
        train_mean_std_path = os.path.join(os.path.dirname(csv_path), f"train_mean_std.txt")
        if phase == 'train':
            mean = self.df['Label'].mean(skipna=True)
            std  = self.df['Label'].std(skipna=True)
            with open(train_mean_std_path, "w") as f:
                f.write(f"{mean:.6f}\n")
                f.write(f"{std:.6f}\n")
        else:
            with open(train_mean_std_path) as f:
                mean = float(f.readline().strip())
                std  = float(f.readline().strip())
        self.df['Label']=(self.df['Label']-mean)/std
        self.lig_graph_dict = load_all_ligand_graph(self.df, ligand_graph_dir, ligand_column=ligand_column)
        self.prot_graph_dict = load_all_protein_multimodal_graph(self.df, protein_graph_dir)
        self.kpgt_dict = load_all_ligand_kpgt(self.df, kpgt_dir, ligand_column=ligand_column)
        self.max_lig_atoms = max_lig_atoms
        self.max_prot_residues = max_prot_residues
        self.protein_column = protein_column
        self.ligand_column = ligand_column
        self.use_pocket = use_pocket
        self.use_assay = use_assay
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        affinity_label = torch.tensor([row['Label']], dtype=torch.float32)
        prot_graph = self.prot_graph_dict[row[self.protein_column]]
        lig_graph = self.lig_graph_dict[row[self.ligand_column]]
       
        protein_length = prot_graph.num_nodes()
        # esm2_feat = self.esm2_dict[row[self.protein_column]]
        # esm2_padded = torch.zeros(self.max_prot_residues, esm2_feat.shape[1], dtype=torch.float32)
        # esm2_padded[:protein_length] = esm2_feat[:self.max_prot_residues]
        kpgt = torch.from_numpy(self.kpgt_dict[row[self.ligand_column]]).float()
        if self.use_pocket:
            pocket_label = torch.from_numpy(np.asarray(list(row['Protein_Pocket']),dtype=int))
            pocket_label_padded = torch.zeros(self.max_prot_residues, dtype=torch.float32)
            pocket_label_padded[:protein_length] = pocket_label[:self.max_prot_residues]
        else:
            pocket_label_padded = torch.tensor([0.0], dtype=torch.float32)
        
        ligand_length = lig_graph.num_nodes()
        mask_prot = torch.zeros(self.max_prot_residues, dtype=torch.bool)
        mask_prot[:protein_length] = True
        mask_lig = torch.zeros(self.max_lig_atoms, dtype=torch.bool)
        mask_lig[:ligand_length] = True

        if not self.use_assay:
            assay_padded = torch.tensor([0.0], dtype=torch.float32)
            mask_assay = torch.tensor([0.0], dtype=torch.float32)
        return prot_graph, kpgt, lig_graph, assay_padded, affinity_label, pocket_label_padded, mask_prot, mask_lig, mask_assay

def collate_fn_PDBbind21_affinity(batch):
    prot_graphs, kpgts, lig_graphs, assays_padded, affinity_labels, pocket_labels_padded, mask_prots, mask_ligs, mask_assays = zip(*batch)
    lig_graphs = dgl.batch(lig_graphs)
    prot_graphs = dgl.batch(prot_graphs)
    kpgts = torch.stack(kpgts, dim=0)
    pocket_labels_padded = torch.stack(pocket_labels_padded, dim=0)
    mask_prots = torch.stack(mask_prots, dim=0)
    mask_ligs = torch.stack(mask_ligs, dim=0)
    affinity_labels = torch.stack(affinity_labels, dim=0)
    mask_assays = torch.stack(mask_assays, dim=0)
    assays_padded = torch.stack(assays_padded, dim=0)
    return prot_graphs, kpgts, lig_graphs, assays_padded, affinity_labels, pocket_labels_padded, mask_prots, mask_ligs, mask_assays


# class PDBBind_Affinity_Dataset(Dataset):
#     def __init__(self, csv_path='dataset/PDBbind/PDBbind21_affinity.csv', 
#                  protein_graph_dir='dataset/protein/dgl_graphs', 
#                  ligand_graph_dir='dataset/ligand/dgl_graphs', 
#                  esm2_dir='dataset/protein/esm2_embedding',
#                  max_lig_atoms=225,
#                  max_prot_residues=1500, 
#                  max_assay_length=100,
#                  dataset_type='train', 
#                  split_dir = 'dataset/PDBbind/split_idx/affinity/',
#                  split_save_dir = 'dataset/PDBbind/split_df/affinity/',
#                  target_types=['Label_IC50_p', 'Label_Kd_p', 'Label_Ki_p'],
#                  split_method='random'
#                  ):
#         split_dir = f"{split_dir}/{split_method}"
#         self.df = pd.read_csv(csv_path, sep='\t')
        # scaler_list = []
        # for col in target_types:
        #     if col in self.df.columns:
        #         mean, std = self.df[col].mean(skipna=True), self.df[col].std(skipna=True)
        #         self.df[col] = (self.df[col]-mean)/std
        #         scaler_list.append([col, mean, std])
        # scaler_df = pd.DataFrame(scaler_list, columns=['Column','Mean','Std'])
        # scaler_df.to_csv(f"{split_dir}/target_mean_std.csv", sep='\t', index=False)
#         self.dataset_type = dataset_type
#         indices = pd.read_csv(f'{split_dir}/{dataset_type}_index.txt', sep='\t', header=None).squeeze().tolist()
#         self.indices = indices
#         self.df = self.df.iloc[indices].reset_index(drop=True)
#         if split_save_dir is not None:
#             self.df.to_csv(f"{split_save_dir}/{dataset_type}.csv", sep='\t', index=False)
#         self.affinity_label_masks=(~self.df[target_types].isna()).astype(int).values.tolist()
#         self.affinity_labels=self.df[target_types].fillna(0).values.tolist()
#         self.lig_graph_dict = load_all_ligand_graphs(self.df, ligand_graph_dir, ligand_column='Ligand_ID')
#         self.prot_graph_dict = load_all_protein_graphs(self.df, protein_graph_dir)
#         self.esm2_dict = load_all_protein_esm2(self.df, esm2_dir)
#         self.max_lig_atoms = max_lig_atoms
#         self.max_prot_residues = max_prot_residues
#         self.max_assay_length = max_assay_length

#         if max_lig_atoms is None or max_prot_residues is None:
#             max_prot_length, max_lig_length, ratio, pos_weight = statistics_of_PDBbind_dataset(csv_path)
#             self.max_lig_atoms, self.max_prot_residues = max_lig_length, max_prot_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         indice = self.indices[idx]
#         esm2_feat = self.esm2_dict[row['Protein_UniProtID']]
#         prot_graph = self.prot_graph_dict[row['Protein_UniProtID']]
#         drug_graph = self.lig_graph_dict[row['Ligand_ID']]
#         pocket_residues = [int(i) + int(row['PDBBind_UniProt_ResStart'])-int(row['PDBBind_PDB_ResStart']) - 1 for i in row['PDBBind_Pocket_ResID'].split('_')]
#         protein_length = esm2_feat.shape[0]
#         ligand_length = drug_graph.num_nodes()
#         esm2_padded = torch.zeros(self.max_prot_residues, esm2_feat.shape[1], dtype=torch.float32)
#         esm2_padded[:protein_length] = esm2_feat[:self.max_prot_residues]
#         pocket_label_padded = torch.zeros(self.max_prot_residues, dtype=torch.float32)
#         for i in pocket_residues:
#             if 0 <= i < protein_length:
#                 pocket_label_padded[i] = 1.0
#         mask_prot = torch.zeros(self.max_prot_residues, dtype=torch.bool)
#         mask_prot[:protein_length] = True
#         mask_lig = torch.zeros(self.max_lig_atoms, dtype=torch.bool)
#         mask_lig[:ligand_length] = True
#         affinity_label, affinity_label_mask = torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)
#         if self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'test':
#             affinity_label = torch.tensor(self.affinity_labels[idx], dtype=torch.float32)
#             affinity_label_mask = torch.tensor(self.affinity_label_masks[idx], dtype=torch.bool)
#         assay_padded, mask_assay = torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)
#         return indice, drug_graph, prot_graph, esm2_padded, assay_padded, affinity_label, affinity_label_mask, pocket_label_padded, ligand_length, protein_length, mask_lig, mask_prot, mask_assay



# def load_all_pocket_label():
#     pass

# class ChEMBL_Affinity_Dataset(Dataset):
#     def __init__(self, csv_path='dataset/ChEMBL/ChEMBL.csv', 
#                  protein_graph_dir='dataset/protein/dgl_graphs', 
#                  ligand_graph_dir='dataset/ligand/dgl_graphs', 
#                  pocket_dir = 'dataset/protein/pockets_padded', 
#                  ligand_column = 'Ligand_ChEMBL_ID', 
#                  esm2_dir='dataset/protein/esm2_embedding',
#                  assay_dir='dataset/assay/biobert',
#                  max_lig_atoms=225,
#                  max_prot_residues=1500, 
#                  max_assay_length=100,
#                  dataset_type='train', 
#                  split_dir = 'dataset/ChEMBL/split_idx/',
#                  target_types=['Label_IC50_p', 'Label_Kd_p', 'Label_Ki_p'],
#                  split_method='random',
#                  use_pocket=False,
#                  use_assay=True,
#                  ):
#         split_dir = f"{split_dir}/{split_method}"
#         self.df = pd.read_csv(csv_path, sep='\t')
#         self.use_pocket = use_pocket
#         self.use_assay = use_assay
#         scaler_list = []
#         for col in target_types:
#             if col in self.df.columns:
#                 mean, std = self.df[col].mean(skipna=True), self.df[col].std(skipna=True)
#                 self.df[col] = (self.df[col]-mean)/std
#                 scaler_list.append([col, mean, std])
#         scaler_df = pd.DataFrame(scaler_list, columns=['Column','Mean','Std'])
#         scaler_df.to_csv(f"{split_dir}/target_mean_std.csv", sep='\t', index=False)
#         self.dataset_type = dataset_type
#         if dataset_type in ['train', 'val', 'test']:
#             indices = pd.read_csv(f'{split_dir}/{dataset_type}_index.txt', sep='\t', header=None).squeeze().tolist()
#             self.indices = indices
#             self.df = self.df.iloc[indices].reset_index(drop=True)
#         else:
#             self.indices = self.df.index
#         self.affinity_label_masks=(~self.df[target_types].isna()).astype(int).values.tolist()
#         self.affinity_labels=self.df[target_types].fillna(0).values.tolist()
#         self.lig_graph_dict = load_all_ligand_graphs(self.df, ligand_graph_dir, ligand_column=ligand_column)
#         self.prot_graph_dict = load_all_protein_graphs(self.df, protein_graph_dir)
#         self.esm2_dict = load_all_protein_esm2(self.df, esm2_dir)
#         self.assay_dict = load_all_assay_biobert(self.df, assay_dir)
#         if use_pocket:
#             self.pocket_labels_padded = load_all_pocket_label(self.df, pocket_dir)
#         self.max_lig_atoms = max_lig_atoms
#         self.max_prot_residues = max_prot_residues
#         self.max_assay_length = max_assay_length
#         self.ligand_column = ligand_column
#         if max_lig_atoms is None or max_prot_residues is None:
#             max_prot_length, max_lig_length, ratio, pos_weight = statistics_of_PDBbind_dataset(csv_path)
#             self.max_lig_atoms, self.max_prot_residues = max_lig_length, max_prot_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         indice = self.indices[idx]
#         esm2_feat = self.esm2_dict[row['Protein_UniProtID']]
#         prot_graph = self.prot_graph_dict[row['Protein_UniProtID']]
#         drug_graph = self.lig_graph_dict[row[self.ligand_column]]
#         protein_length = esm2_feat.shape[0]
#         ligand_length = drug_graph.num_nodes()
#         esm2_padded = torch.zeros(self.max_prot_residues, esm2_feat.shape[1], dtype=torch.float32)
#         esm2_padded[:protein_length] = esm2_feat[:self.max_prot_residues]
#         mask_prot = torch.zeros(self.max_prot_residues, dtype=torch.bool)
#         mask_prot[:protein_length] = True
#         mask_lig = torch.zeros(self.max_lig_atoms, dtype=torch.bool)
#         mask_lig[:ligand_length] = True

#         affinity_label, affinity_label_mask = torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)
#         if self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'test':
#             affinity_label = torch.tensor(self.affinity_labels[idx], dtype=torch.float32)
#             affinity_label_mask = torch.tensor(self.affinity_label_masks[idx], dtype=torch.bool)

#         assay_padded, mask_assay, pocket_label_padded = torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)
#         if self.use_pocket:
#             pocket_label_padded = self.pocket_labels_padded[f"{row['Protein_UniProtID']}_{row[self.ligand_column]}"]
#         if self.use_assay:
#             assay_feat = self.assay_dict[row['Assay_ChEMBL_ID']]
#             assay_length = assay_feat.shape[0]
#             assay_padded = torch.zeros(self.max_assay_length, assay_feat.shape[1], dtype=torch.float32)
#             assay_padded[:assay_length] = assay_feat[:self.max_assay_length]
#             mask_assay = torch.zeros(self.max_assay_length, dtype=torch.bool)
#             mask_assay[:assay_length] = True  
#         return indice, drug_graph, prot_graph, esm2_padded, assay_padded, affinity_label, affinity_label_mask, pocket_label_padded, ligand_length, protein_length, mask_lig, mask_prot, mask_assay

# def collate_fn_PDBBind_affinity(batch):
#     indices, drug_graphs, prot_graphs, esm2_feats, assay_feats, affinity_labels, affinity_label_masks, pocket_labels_padded, lig_length,  prot_length, mask_ligs, mask_prots, mask_assays = zip(*batch)
#     indices = torch.tensor(indices, dtype=torch.long)
#     drug_graphs = dgl.batch(drug_graphs)
#     prot_graphs = dgl.batch(prot_graphs)
#     esm2_feats = torch.stack(esm2_feats, dim=0)
#     assay_feats = torch.stack(assay_feats, dim=0)
#     affinity_labels = torch.stack(affinity_labels, dim=0)
#     affinity_label_masks = torch.stack(affinity_label_masks, dim=0)
#     pocket_labels_padded = torch.stack(pocket_labels_padded, dim=0)
#     protein_lengths = torch.tensor(prot_length, dtype=torch.long)
#     ligand_lengths = torch.tensor(lig_length, dtype=torch.long)
#     mask_prots = torch.stack(mask_prots, dim=0)
#     mask_ligs = torch.stack(mask_ligs, dim=0)
#     mask_assays = torch.stack(mask_assays, dim=0)
#     return indices, drug_graphs, prot_graphs, esm2_feats, assay_feats, affinity_labels, affinity_label_masks, pocket_labels_padded, ligand_lengths, protein_lengths, mask_ligs, mask_prots, mask_assays

if __name__ == "__main__":
    # max_prot_length, max_lig_length, max_pocket_length, ratio, pos_weight = statistics_of_PDBbind_dataset(csv_path='dataset/PDBbind/PDBbind21_pocket.csv')
    # print(max_prot_length, max_lig_length, max_pocket_length, ratio, pos_weight)
    # max_prot_length, max_lig_length, max_pocket_length, ratio, pos_weight = statistics_of_PDBbind_dataset(csv_path='dataset/PDBbind/PDBbind21_affinity.csv')
    # print(max_prot_length, max_lig_length, max_pocket_length, ratio, pos_weight)
    # dataset = Affinity_Dataset(csv_path='dataset/PDBbind/PDBbind21_affinity.csv', dataset_type='train')
    # dataset = PDBbind21_Pocket_Dataset(csv_path='dataset/PDBbind/PDBbind21_pocket.csv', dataset_type='train')

    # dataset = ChEMBL_Affinity_Dataset(csv_path='dataset/ChEMBL/ChEMBL.csv', ligand_column='Ligand_ChEMBL_ID', dataset_type='test', split_dir = 'dataset/ChEMBL/split/')


    # dataset = ChEMBL_Affinity_Dataset(csv_path='dataset/ChEMBL/ChEMBL.csv', ligand_column='Ligand_ChEMBL_ID', dataset_type='test', split_dir = 'dataset/ChEMBL/split/')
    # print(statistics_of_ChEMBL_dataset())

    # a =  PDBbind21_pocket_Dataset('dataset/dataset/PDBbind21_pocket/unseen_protein/train.csv')[0]

    pass