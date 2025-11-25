import os
import re
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
import dgl
from dgl import DGLGraph
import dgllife

def MorganFingerPrint_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))




def get_atom_features(atom):
    feats = []
    feats += one_of_k_encoding_unk(atom.GetSymbol(),
        ['C','N','O','F','P','Cl','Br','I','Si','DU'])
    feats += one_of_k_encoding_unk(atom.GetImplicitValence(),
        [0,1,2,3,'UNK'])
    feats += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(),
        [0,1,'UNK'])
    feats += one_of_k_encoding_unk(atom.GetDegree(),
        [0,1,2,3,4,5,'UNK'])
    feats += one_of_k_encoding_unk(atom.GetFormalCharge(),
        [-1,0,1,'UNK'])
    feats += one_of_k_encoding_unk(atom.GetHybridization(),
        [Chem.rdchem.HybridizationType.SP,
         Chem.rdchem.HybridizationType.SP2,
         Chem.rdchem.HybridizationType.SP3,
         Chem.rdchem.HybridizationType.SP3D,
         'UNK'])
    feats += one_of_k_encoding_unk(atom.GetChiralTag(),
        [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
         'UNK'])
    feats.append(1 if atom.GetIsAromatic() else 0)
    feats.append(1 if atom.IsInRing() else 0)
    return np.array(feats)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def Graph_smiles(smiles, node_only=True):
    molecule = Chem.MolFromSmiles(smiles)
    g = DGLGraph()
    g.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_features(atom_i) 
        node_features.append(atom_i_features)
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                g.add_edges(i,j)
                if not node_only:
                    bond_features_ij = get_bond_features(bond_ij) 
                    edge_features.append(bond_features_ij)
    g.ndata['atom_feat'] = torch.from_numpy(np.array(node_features)).float()
    if not node_only:
        g.edata['bond_feat'] = torch.from_numpy(np.array(edge_features)).float()
    return g

def generate_all_ligand_morgan(df_path='dataset/dataset/PDBbind/PDBbind21_pocket.csv', 
                               fp_dir="dataset/public/ligand/morgan"):
    os.makedirs(fp_dir, exist_ok=True)
    df = pd.read_csv(df_path, sep="\t")
    for index, row in df.iterrows():
        if os.path.exists(f"{fp_dir}/{row['Ligand_ID']}.txt"):
            continue
        np.savetxt(f"{fp_dir}/{row['Ligand_ID']}.txt", MorganFingerPrint_smiles(row['Ligand_SMILES']), fmt='%d')


def generate_all_ligand_graph(df_path='dataset/PDBbind/PDBbind21_pocket.csv', 
                              graph_dir="dataset/public/ligand/graphs", ligand_column='Ligand_ID', smiles_column='Ligand_SMILES'):
    os.makedirs(graph_dir, exist_ok=True)
    df = pd.read_csv(df_path, sep="\t")
    for _, row in df.iterrows():
        if os.path.exists(f"{graph_dir}/{row[ligand_column]}.dgl"):
            continue 
        g = Graph_smiles(row[smiles_column], node_only=True)
        dgl.save_graphs(f"{graph_dir}/{row[ligand_column]}.dgl", [g])

def load_all_ligand_graph(df, graph_dir="dataset/public/ligand/graphs", ligand_column='Ligand_CID'):
    graph_dict = {}
    ligand_ids = df[ligand_column].unique()
    for ligand_id in ligand_ids:
        graphs, _ = dgl.load_graphs(f"{graph_dir}/{ligand_id}.dgl")
        graph_dict[ligand_id] = graphs[0]
    return graph_dict

def load_all_ligand_kpgt(df, kpgt_dir="dataset/public/ligand/kpgt", ligand_column='Ligand_ID'):
    kpgt_dict = {}
    ligand_ids = df[ligand_column].unique()
    for ligand_id in ligand_ids: 
        kpgt_dict[ligand_id] = np.loadtxt(f"{kpgt_dir}/{ligand_id}.txt")
    return kpgt_dict



if __name__ == "__main__":
    # generate_all_ligand_graph(df_path=f'dataset/processed/Davis/all.csv', ligand_column='Ligand_CID', smiles_column='Ligand_SMILES')
    # generate_all_ligand_graph(df_path='/home/hy/data/protein-ligand/protein-ligand/project/PLMCA/dataset/raw/ChEMBL/ChEMBL.csv', ligand_column='Ligand_ChEMBL_ID', smiles_column='Smiles')

    generate_all_ligand_graph(df_path=f'task/5SQQ/5SQQ.csv',  graph_dir="task/5SQQ/ligand/graphs", ligand_column='Ligand_ID', smiles_column='Ligand_SMILES')
