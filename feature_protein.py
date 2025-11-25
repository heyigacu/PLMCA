import os
import requests
import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import get_surface
import periodictable
from scipy.spatial import cKDTree
from tools.esm2.esm_embedding import generate_esm_embeddings
from Bio import PDB, SeqIO
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

DSSP = './tools/dssp'
MSMS = './tools/msms'
ANKH_PATH = './tools/ankh-large',
##############
# seq and pbd
##############

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        seq = ''.join(lines[1:])
    else:
        seq = None  
    return seq

def download_alphafold_structure(uniprot_id, save_dir="pdbs"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{uniprot_id}.pdb")
    if os.path.exists(file_path):
        return None
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return None
    else:
        print(f"Failed to download structure for UniProt ID: {uniprot_id}")
        return uniprot_id

def fetch_all_sequence(df_path='dataset/PDBbind/PDBbind21_pocket.csv', 
                         fasta_dir = "dataset/protein/fastas", 
                         protein_column='Protein_UniProtID'):
    os.makedirs(fasta_dir,exist_ok=True)
    df=pd.read_csv(df_path,sep='\t')
    none_ls = []
    for _,row in df.iterrows():
        uniprot_id=row[protein_column]
        if not os.path.exists(f"{fasta_dir}{uniprot_id}.fasta"):
            none_ls.append(uniprot_id)
    none_ls = list(set(none_ls))
    error_uniprot_ids = []
    with open(f"{fasta_dir}/{uniprot_id}.fasta", "w") as f:
        for uniprot_id in none_ls:
            seq = fetch_uniprot_sequence(uniprot_id)
            if seq is not None:
                f.write(f">{uniprot_id}\n{seq}\n")
            else:
                error_uniprot_ids.append(uniprot_id)
    with open(f'seq_none.txt', 'w') as f:
        for item in error_uniprot_ids:
            f.write(f"{item}\n")
    print(error_uniprot_ids)

import shutil
def download_all_alphafold_structure(df_path='dataset/PDBbind/PDBbind21_pocket.csv', 
                                       pdb_dir='dataset/protein/pdbs/', 
                                       protein_column='Protein_UniProtID'):
    df=pd.read_csv(df_path,sep='\t')
    uniprot_ids = df[protein_column].unique()
    need_uniprot_ids=[]
    for uniprot_id in uniprot_ids:
        # if not os.path.exists(os.path.join(f"dataset/feature/protein/pdbs/{uniprot_id}.pdb")):
        #     shutil.copy(f"dataset/feature/protein/pdbs/{uniprot_id}.pdb", os.path.join(f"{pdb_dir}/{uniprot_id}.pdb"))
        #     continue
        if not os.path.exists(os.path.join(f"{pdb_dir}/{uniprot_id}.pdb")):
            need_uniprot_ids.append(uniprot_id)
    print(len(need_uniprot_ids))
    error_uniprot_ids = []
    for uniprot_id in need_uniprot_ids:
        rst = download_alphafold_structure(uniprot_id, pdb_dir)
        if rst is not None:
            error_uniprot_ids.append(rst)
    with open(f'pdb_none.txt', 'w') as f:
        for item in error_uniprot_ids:
            f.write(f"{item}\n")
    print(error_uniprot_ids)

def process_pdb_for_dssp(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()
    cryst1_lines = [line for line in lines if line.startswith("CRYST1")]
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("MODEL"):
            start_idx = i
            break
    with open(output_path, "w") as out:
        for line in cryst1_lines:
            out.write(line)
        for line in lines[start_idx:]:
            out.write(line)

def process_all_pdb_for_dssp(df_path='dataset/PDBbind/PDBbind21_pocket.csv', 
                            input_pdb_dir='dataset/feature/protein/pdbs/', 
                            save_pdb_dir='dataset/feature/protein/pdbs_noheader/'):
    os.makedirs(save_pdb_dir, exist_ok=True)
    df=pd.read_csv(df_path,sep='\t')
    uniprot_ids = df['Protein_UniProtID'].unique()
    for uniprot_id in uniprot_ids:
        if not os.path.exists(f"{save_pdb_dir}/{uniprot_id}.pdb"):
            process_pdb_for_dssp(f"{input_pdb_dir}/{uniprot_id}.pdb", f"{save_pdb_dir}/{uniprot_id}.pdb")



from Bio.PDB import PDBParser, PPBuilder
def pdb_to_seq(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_file)
    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure)
    if len(peptides) == 0:
        return ""
    seq = str(peptides[0].get_sequence())
    return seq

##############
# ankh
##############



def generate_all_protein_ankh(df_path, ankh_path='./tools/ankh-large',device='cuda',ankh_dir='./dataset/feature/protein/ankh/'):
    df = pd.read_csv(df_path, sep="\t")
    os.makedirs(ankh_dir, exist_ok=True)
    # uniprot_ids = df['Protein_UniProtID'].unique()
    # sequences = [pdb_to_seq(f'dataset/public/protein/pdbs/{uniprot_id}.pdb') for uniprot_id in uniprot_ids]
    # tuple_ls = zip(uniprot_ids, sequences)
    tuple_ls = list(set(zip(df['Protein_UniProtID'], df['Protein_Sequence'])))
    
    tokenizer = AutoTokenizer.from_pretrained(ankh_path)
    model = T5EncoderModel.from_pretrained(ankh_path)
    model.to(device)
    model.eval()
    for uniprot_id,sequence in tuple_ls:
        if os.path.exists(f'{ankh_dir}/{uniprot_id}.npy'):
            continue
        ids = tokenizer.batch_encode_plus([list(sequence)], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state[0,:len(sequence)].cpu().numpy()
            print(emb.shape, f'{ankh_dir}/{uniprot_id}.npy')
            np.save(f'{ankh_dir}/{uniprot_id}.npy',emb)


def load_all_protein_ankh(df, ankh_dir="dataset/feature/protein/ankh"):
    ankh_dict = {}
    uniprot_ids = df["Protein_UniProtID"].unique()
    for uniprot_id in uniprot_ids:
        ankh_dict[uniprot_id] = torch.tensor(np.load(f"{ankh_dir}/{uniprot_id}.npy"),dtype=torch.float32) 
    return ankh_dict


##############
# esm2
##############

def generate_all_protein_esm(df_path='dataset/PDBbind/PDBbind21_pocket.csv', 
                                  save_folder = "dataset/feature/protein/esm2_embedding/",
                                  batch_size=2):
    df = pd.read_csv(df_path, sep="\t")
    # uniprot_ids = df['Protein_UniProtID'].unique()
    # sequences = [pdb_to_seq(f'dataset/public/protein/pdbs/{uniprot_id}.pdb') for uniprot_id in uniprot_ids]
    # tuple_ls = zip(uniprot_ids, sequences)
    os.makedirs(save_folder, exist_ok=True)
    need_tuple_ls = []
    
    tuple_ls = list(set(zip(df['Protein_UniProtID'], df['Protein_Sequence'])))
    for uniprot_id,sequence in tuple_ls:
        if not os.path.exists(f'{save_folder}/{uniprot_id}.npz'):
            need_tuple_ls.append((uniprot_id,sequence))
    if len(need_tuple_ls) > 0:
        generate_esm_embeddings(need_tuple_ls, 'esm2_t6_8M_UR50D', batch_size, save_dir=save_folder)


def load_all_protein_esm2(df, esm2_dir="dataset/feature/protein/esm2"):
    esm2_dict = {}
    uniprot_ids = df["Protein_UniProtID"].unique()
    for uniprot_id in uniprot_ids:
        esm2_dict[uniprot_id] = torch.tensor(np.load(f"{esm2_dir}/{uniprot_id}.npz")['data'],dtype=torch.float32) 
    return esm2_dict


##############
# physicochemical
##############
"""
combined MMFuncPhos and CAPLA:
    MMFuncPhos: A Multi-Modal Learning Framework for Identifying Functional Phosphorylation Sites and Their Regulatory Types
    CAPLA: improved prediction of protein–ligand binding affinity by a deep learning approach based on a cross-attention mechanism 
"""
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
SS_LIST = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
map_aa_number = {aa: i for i, aa in enumerate(AA_LIST)}
map_ss_dssp = {ss: i for i, ss in enumerate(SS_LIST)}
map_aa_phy = {
    'C': 0,
    'D': 1, 'E': 1,
    'R': 2, 'K': 2,
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,
    'Y': 4, 'M': 4, 'T': 4, 'S': 4,
    'I': 5, 'L': 5, 'F': 5, 'P': 5,
    'A': 6, 'G': 6, 'V': 6,
}
map_aa_chem = {
    'A': 0,'V':0,'L':0,'I':0,'M':0,
    'F':0,'W':0,'P':0,'G':0,'C':0,
    'S':1,'T':1,'N':1,'Q':1,'Y':1,
    'D':2,'E':2,
    'K':3,'R':3,'H':3
}
SANDER_MAX_ASA = {
    'A': 106, 'R': 248, 'N': 157, 'D': 163, 'C': 135, 'Q': 198, 'E': 194,
    'G': 84,  'H': 184, 'I': 169, 'L': 164, 'K': 205, 'M': 188, 'F': 197,
    'P': 136, 'S': 130, 'T': 142, 'W': 227, 'Y': 222, 'V': 142
}

def one_hot(idx, dim):
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v

def get_bfactors(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    bfactors = []
    for model in structure:
        for chain in model:
            for res in chain:
                bfactors.append(res['CA'].get_bfactor())
    return bfactors

def cal_protein_phychem_feat(dssp_rst_path, pdb_path):
    # dim 42
    bfactors = get_bfactors(pdb_path)
    with open(dssp_rst_path, 'r') as fr:
        dssp_data = fr.readlines()
    seq_feature = []
    for idx,i in enumerate(range(25, len(dssp_data))):
        line = dssp_data[i]
        aa = line[13].upper() if line[13] != '!' else 'X'
        ss = line[16] if line[16] in SS_LIST else ' '
        rasa = int(line[35:38].strip())/SANDER_MAX_ASA[aa]
        rasa = rasa if rasa <1 else 1.
        f_aa_number = one_hot(map_aa_number.get(aa, 21), 21)
        f_aa_chem  = one_hot(map_aa_chem.get(aa, 0), 4)
        f_aa_phy = one_hot(map_aa_phy.get(aa, 6), 7)
        f_ss_dssp  = one_hot(map_ss_dssp.get(ss, 7), 8)
        bfactor = bfactors[idx]/100
        feature = np.concatenate([f_aa_number, f_aa_chem, f_aa_phy, f_ss_dssp, np.array([rasa, bfactor])])
        seq_feature.append(feature)
    seq_feature = np.array(seq_feature, dtype=np.float32)
    return seq_feature

def generate_all_protein_phychem(df_path='dataset/feature/PDBbind/Ki/random/all.csv', 
                              dssp_dir="dataset/feature/protein/dssp", 
                              pdb_noheader_dir="dataset/feature/protein/pdbs_noheader", 
                              phychem_dir="dataset/feature/protein/phychem"):
    df = pd.read_csv(df_path, sep="\t")
    uniprot_ids = df['Protein_UniProtID'].unique()
    os.makedirs(dssp_dir, exist_ok=True)
    for uniprot_id in uniprot_ids:
        if os.path.exists(f"{dssp_dir}/{uniprot_id}.dssp"):
            continue
        os.system(f"{DSSP} -i {pdb_noheader_dir}/{uniprot_id}.pdb -o {dssp_dir}/{uniprot_id}.dssp")
    os.makedirs(phychem_dir, exist_ok=True)
    for uniprot_id in uniprot_ids:
        if os.path.exists(f"{phychem_dir}/{uniprot_id}.npz"):
            continue
        phychem_feat = cal_protein_phychem_feat(f'{dssp_dir}/{uniprot_id}.dssp', f"{pdb_noheader_dir}/{uniprot_id}.pdb")
        np.savez(f'{phychem_dir}/{uniprot_id}.npz', data=phychem_feat)


# def load_all_protein_phychem(df, dssp_dir="dataset/feature/protein/phychem"):
#     feat_dict = {}
#     tuple_ls = list(set(zip(df['Protein_UniProtID'], df['Sequence'])))
#     for uniprot_id, sequence in tuple_ls:
#         prot_emb = np.load(os.path.join(dssp_dir, f"{uniprot_id}.npz"))['feat']
#         feat_dict[uniprot_id] = prot_emb
#     return feat_dict


##############
# local graph
##############

def cal_protein_MSMS(uniprot_id, 
                     pdb_dir="dataset/feature/protein/pdbs_noheader", 
                     save_dir="dataset/feature/protein/msms", ):
    pdb_file = f"{pdb_dir}/{uniprot_id}.pdb"
    save_file = f"{save_dir}/{uniprot_id}.npy"
    parser = PDBParser(QUIET=True)
    X, chain_atom = [], ['N', 'CA', 'C', 'O']
    model = parser.get_structure('model', pdb_file)[0]
    chain = next(model.get_chains())
    try:
        surf = get_surface(chain, MSMS=MSMS)
        surf_tree = cKDTree(surf)
    except:
        surf = np.empty(0)
    for residue in chain:
        line = []
        atoms_coord = np.array([atom.get_coord() for atom in residue])
        if surf.size == 0:
            dist, _ = surf_tree.query(atoms_coord)
            closest_pos = atoms_coord[np.argmin(dist)]
        else:
            closest_pos = atoms_coord[-1]
        atoms = list(residue.get_atoms())
        ca_pos = residue['CA'].get_coord() if 'CA' in residue else residue.child_list[0].get_coord()
        pos_s = un_s = 0
        for atom in atoms:
            if atom.name in chain_atom:
                line.append(atom.get_coord())
            else:
                pos_s += calMass(atom, True)
                un_s += calMass(atom, False)
        if len(line) != 4:
            line += [list(ca_pos)] * (4 - len(line))
        R_pos = ca_pos if un_s == 0 else pos_s / un_s
        line.append(R_pos)
        line.append(closest_pos)
        X.append(line)
    np.save(save_file, X)

def generate_all_protein_msms(df_path='dataset/feature/PDBbind/Ki/random/all.csv', 
                              msms_dir="dataset/feature/protein/dssp", 
                              pdb_dir="dataset/feature/protein/pdbs", ):
    df = pd.read_csv(df_path, sep="\t")
    uniprot_ids = df['Protein_UniProtID'].unique()
    os.makedirs(msms_dir, exist_ok=True)
    for uniprot_id in uniprot_ids:
        if os.path.exists(f"{msms_dir}/{uniprot_id}.npy"):
            continue
        cal_protein_MSMS(uniprot_id, pdb_dir, msms_dir)

def lowerElem(elem):
    if len(elem) == 1:
        return elem
    return elem[0] + elem[1].lower()

def calMass(atom,pos=True):
    if pos:
        return periodictable.elements.symbol(lowerElem(atom.element)).mass * np.array(atom.get_coord())
    return periodictable.elements.symbol(lowerElem(atom.element)).mass

def _quaternions_from_R(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    mags = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)))
    def _R(i, j): return R[..., i, j]
    signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
    xyz = signs * mags
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    return F.normalize(Q, dim=-1)

def batch_local_graph(xyz, mask, top_k=30, rbf_num=8, D_max=20.):
    """
    https://dl.acm.org/doi/10.5555/3454287.3455704
    """
    B, N, _, _ = xyz.shape
    device = xyz.device
    CaX = xyz[:, :, 1]
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
    dX = CaX.unsqueeze(1) - CaX.unsqueeze(2)
    D = mask_2D * torch.sqrt((dX ** 2).sum(-1) + 1e-6)
    D_max_val, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max_val
    _, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
    A_n, A_ca, A_c = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    u = F.normalize(A_n - A_ca, dim=-1)
    v = F.normalize(A_ca - A_c, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v, dim=-1), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=-1)
    t_nodes = F.normalize(xyz[:, :, [0, 2, 3, 4, 5]] - A_ca.unsqueeze(-2), dim=-1)
    node_dir = torch.matmul(t_nodes, local_frame).flatten(-2)
    X_ca_chain = xyz[:, :, :3].reshape(B, 3 * N, 3)
    dX_chain = X_ca_chain[:, 1:] - X_ca_chain[:, :-1]
    U = F.normalize(dX_chain, dim=-1)
    u_2, u_1, u_0 = U[:, :-2], U[:, 1:-1], U[:, 2:]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cosD = torch.sum(n_2 * n_1, -1).clamp(-1 + 1e-7, 1 - 1e-7)
    D_angles = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D_angles = F.pad(D_angles, [1, 2]).reshape(B, N, 3)
    dihedral = torch.cat([torch.cos(D_angles), torch.sin(D_angles)], -1)
    cosD2 = (u_2 * u_1).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    D_bond = torch.acos(cosD2)
    D_bond = F.pad(D_bond, [1, 2]).reshape(B, N, 3)
    bond_angles = torch.cat([torch.cos(D_bond), torch.sin(D_bond)], -1)
    node_angle = torch.cat([dihedral, bond_angles], -1)
    rel_list = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4], [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]]
    D_node = torch.norm(xyz[:, :, rel_list[0]] - xyz[:, :, rel_list[1]], dim=-1)
    D_mu_node = torch.linspace(0, D_max, rbf_num, device=device).view(1, 1, 1, -1)
    D_sigma = D_max / rbf_num
    node_rbf = torch.exp(-((D_node.unsqueeze(-1) - D_mu_node) / D_sigma) ** 2).flatten(-2)
    node_feat = torch.cat([node_dir, node_angle, node_rbf], -1)
    K = E_idx.shape[-1]
    X_expand = xyz.unsqueeze(2).expand(-1, -1, K, -1, -1)
    gather_idx = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, xyz.shape[2], xyz.shape[3])
    X_neigh = X_expand.gather(1, gather_idx)
    Ca_expand = A_ca.unsqueeze(2).unsqueeze(3)
    t_edge = F.normalize(X_neigh - Ca_expand, dim=-1)
    edge_dir = torch.matmul(t_edge, local_frame.unsqueeze(2)).flatten(-2)
    D_edge = torch.norm(X_neigh - Ca_expand, dim=-1)
    D_mu_edge = torch.linspace(0, D_max, rbf_num, device=device).view(1, 1, 1, 1, -1)
    edge_rbf = torch.exp(-((D_edge.unsqueeze(-1) - D_mu_edge) / D_sigma) ** 2).flatten(-2)
    dX_ca = CaX[:, 1:, :] - CaX[:, :-1, :]
    U_ca = F.normalize(dX_ca, dim=-1)
    u2 = U_ca[:, :-2]; u1 = U_ca[:, 1:-1]
    n2 = F.normalize(torch.cross(u2, u1, dim=-1), dim=-1)
    o1 = F.normalize(u2 - u1, dim=-1)
    O = torch.stack((o1, n2, torch.cross(o1, n2, dim=-1)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0, 0, 1, 2), 'constant', 0)
    batch_idx = torch.arange(B, device=device).unsqueeze(-1).unsqueeze(-1).expand(B, N, K)
    neigh_idx = E_idx.long()
    O_neighbors = O[batch_idx, neigh_idx, :].view(B, N, K, 9)
    X_neighbors = CaX[batch_idx, neigh_idx, :].view(B, N, K, 3)
    O_mat = O.view(B, N, 3, 3)
    O_neighbors_mat = O_neighbors.view(B, N, K, 3, 3)
    dX_rel = X_neighbors - CaX.unsqueeze(2)
    dU = torch.matmul(O_mat.unsqueeze(2), dX_rel.unsqueeze(-1)).squeeze(-1)
    dU = F.normalize(dU, dim=-1)
    R = torch.matmul(O_mat.unsqueeze(2).transpose(-1, -2), O_neighbors_mat)
    Q = _quaternions_from_R(R)
    edge_ori = torch.cat([dU, Q], -1)
    edge_feat_all = torch.cat([edge_dir, edge_ori, edge_rbf], -1)
    B, N, K, C = edge_feat_all.shape
    edge_feat_all = edge_feat_all.reshape(B, N * K, C)
    graphs = []
    for b in range(B):
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
        dst = E_idx[b].reshape(-1).long()
        g = dgl.graph((src, dst), num_nodes=N)
        g.ndata['local'] = node_feat[b]
        g.edata['local'] = edge_feat_all[b]
        graphs.append(g)
    return graphs[0]

##############
# total graph
##############

def calculate_all_protein_multimodal_feature(df_path='dataset/feature/PDBbind/Ki/random/all.csv',
                    pdb_noheader_dir="dataset/feature/protein/pdbs_noheader", 
                    msms_dir="dataset/feature/protein/msms", 
                    dssp_dir="dataset/feature/protein/dssp", 
                    phychem_dir="dataset/feature/protein/phychem", 
                    esm2_dir="dataset/feature/protein/esm2",
                    ankh_dir='./dataset/feature/protein/ankh/'):
    generate_all_protein_ankh(df_path, device='cuda:1',ankh_dir=ankh_dir)
    generate_all_protein_esm(df_path=df_path, save_folder=esm2_dir,  batch_size=1)
    generate_all_protein_phychem(df_path=df_path, dssp_dir=dssp_dir, pdb_noheader_dir=pdb_noheader_dir, phychem_dir=phychem_dir)
    generate_all_protein_msms(df_path=df_path, msms_dir=msms_dir, pdb_dir=pdb_noheader_dir)                



def calculate_protein_multimodal_graph(uniprot_id='A0A0F7UUA6', 
                                 msms_dir="dataset/feature/protein/msms", 
                                 phychem_dir="dataset/feature/protein/phychem", 
                                 esm2_dir="dataset/feature/protein/esm2",
                                 ankh_dir="dataset/feature/protein/ankh",
                                 top_k=30, rbf_num=8,):
    coords = torch.from_numpy(np.load(f"{msms_dir}/{uniprot_id}.npy"))
    protein_length = coords.shape[0]
    g = batch_local_graph(coords.unsqueeze(0), torch.ones(protein_length).float().unsqueeze(0), top_k=top_k, rbf_num=rbf_num, D_max=20.)
    esm2_data=torch.from_numpy(np.load(f"{esm2_dir}/{uniprot_id}.npz")['data'])
    phychem_data=torch.from_numpy(np.load(f"{phychem_dir}/{uniprot_id}.npz")['data'])
    ankh_data=torch.from_numpy(np.load(f"{ankh_dir}/{uniprot_id}.npy"))
    print(uniprot_id, protein_length, esm2_data.shape, ankh_data.shape, phychem_data.shape)
    g.ndata['esm2'] = esm2_data
    g.ndata['ankh'] = ankh_data
    g.ndata['phychem'] = phychem_data
    return g

def delete_feature(uniprot_id='A0A0F7UUA6', 
                msms_dir="dataset/public/protein/msms", 
                phychem_dir="dataset/public/protein/phychem", 
                dssp_dir="dataset/public/protein/dssp", 
                esm2_dir="dataset/public/protein/esm2",
                ankh_dir="dataset/public/protein/ankh",
                pdb_noheader_dir="dataset/public/protein/pdbs_noheader",
                ):
    os.remove(f"{pdb_noheader_dir}/{uniprot_id}.pdb")
    os.remove(f"{msms_dir}/{uniprot_id}.npy")
    os.remove(f"{dssp_dir}/{uniprot_id}.dssp")
    os.remove(f"{phychem_dir}/{uniprot_id}.npz")
    os.remove(f"{esm2_dir}/{uniprot_id}.npz")
    os.remove(f"{ankh_dir}/{uniprot_id}.npy")
   

def generate_all_protein_multimodal_graph(df_path='dataset/feature/PDBbind/Ki/random/all.csv', 
                              esm2_dir="dataset/feature/protein/esm2", 
                              msms_dir="dataset/feature/protein/msms", 
                              phychem_dir="dataset/feature/protein/phychem",
                              ankh_dir="dataset/feature/protein/ankh", 
                              save_dir="dataset/feature/protein/mm_graphs",
                              top_k=30, rbf_num=8,):
    df = pd.read_csv(df_path, sep="\t")
    uniprot_ids = df['Protein_UniProtID'].unique()
    erros = []
    for uniprot_id in uniprot_ids:
        if os.path.exists(f"{save_dir}/{uniprot_id}.dgl"):
            continue
        # try:
        g = calculate_protein_multimodal_graph(uniprot_id, msms_dir, phychem_dir, esm2_dir, ankh_dir, top_k=top_k, rbf_num=rbf_num)
        dgl.save_graphs(os.path.join(save_dir,f"{uniprot_id}.dgl"),[g])
        # except:
        #     erros.append(uniprot_id)
    print(erros)


def load_all_protein_multimodal_graph(df, graph_dir="dataset/feature/protein/mm_graphs"):
    graph_dict = {}
    uniprot_ids = df['Protein_UniProtID'].unique()
    for uniprot_id in uniprot_ids:
        file_path = os.path.join(graph_dir, f"{uniprot_id}.dgl")
        graphs, _ = dgl.load_graphs(file_path)
        graph_dict[uniprot_id] = graphs[0]
    return graph_dict


def process_train_dataset(dataset_name = 'PDBbind21_pocket', use_public=True):
    df_path = f"dataset/processed/{dataset_name}/all.csv"
    if use_public:
        feature_dir = f"dataset/public/"
    else:
        feature_dir = f"dataset/processed/{dataset_name}/feature/"
    pdb_dir = f"{feature_dir}/protein/pdbs/"
    pdb_noheader_dir = f"{feature_dir}/protein/pdbs_noheader/"
    msms_dir=f"{feature_dir}/protein/msms"
    dssp_dir=f"{feature_dir}/protein/dssp"
    phychem_dir=f"{feature_dir}/protein/phychem"
    esm2_dir=f"{feature_dir}/protein/esm2"
    ankh_dir=f'{feature_dir}/protein/ankh/'
    save_dir=f'{feature_dir}/protein/mm_graphs/'
    download_all_alphafold_structure(df_path, pdb_dir)
    process_all_pdb_for_dssp(df_path, pdb_dir, pdb_noheader_dir)    
    calculate_all_protein_multimodal_feature(df_path, pdb_noheader_dir, msms_dir, dssp_dir, phychem_dir, esm2_dir, ankh_dir)
    generate_all_protein_multimodal_graph(df_path, esm2_dir, msms_dir, phychem_dir, ankh_dir, save_dir)


def process_predict_dataset(dataset_name = '5SQQ', use_public=False):
    df_path = f"task/{dataset_name}/{dataset_name}.csv"
    if use_public:
        feature_dir = f"dataset/public/"
    else:
        feature_dir = f"task/{dataset_name}/"
    pdb_dir = f"{feature_dir}/protein/pdbs/"
    pdb_noheader_dir = f"{feature_dir}/protein/pdbs_noheader/"
    msms_dir=f"{feature_dir}/protein/msms"
    dssp_dir=f"{feature_dir}/protein/dssp"
    phychem_dir=f"{feature_dir}/protein/phychem"
    esm2_dir=f"{feature_dir}/protein/esm2"
    ankh_dir=f'{feature_dir}/protein/ankh/'
    save_dir=f'{feature_dir}/protein/mm_graphs/'

    process_all_pdb_for_dssp(df_path, pdb_dir, pdb_noheader_dir)    
    calculate_all_protein_multimodal_feature(df_path, pdb_noheader_dir, msms_dir, dssp_dir, phychem_dir, esm2_dir, ankh_dir)
    generate_all_protein_multimodal_graph(df_path, esm2_dir, msms_dir, phychem_dir, ankh_dir, save_dir)


if __name__ == '__main__':
    # process_train_dataset('Davis')
    # process_train_dataset('KiBA')
    # process_train_dataset('ChEMBL')
    process_predict_dataset('5SQQ')
