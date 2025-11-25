

import os
import subprocess
import pandas as pd

import subprocess
import re
import os

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import shutil
import pandas as pd

VINA = 'vina'
OBABEL = 'obabel'
mgltools_install_dir = '/home/hy/Softwares/MGLTools/'
# Prepare ligand and receptor scripts paths
prepare_ligand4_path = os.path.join(mgltools_install_dir, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
prepare_receptor4_path = os.path.join(mgltools_install_dir, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py')
pythonsh_path = os.path.join(mgltools_install_dir, 'bin/pythonsh')
parent_dir = os.path.abspath(os.path.dirname(__name__))

def generate_pdbqt(df, lig_mol2_save_dir, lig_pdbqt_save_dir, rec_pdb_dir, rec_pdbqt_save_dir):
    lig_ls = []
    rec_ls = []
    for index, row in df.iterrows():
        pdb_name = row['Protein_UniProtID']
        lig_name = row['Ligand_ID']
        smiles = row['Ligand_SMILES']
        lig_mol2_path = os.path.join(lig_mol2_save_dir, lig_name+'.mol2')
        rec_pdb_path = os.path.join(rec_pdb_dir, pdb_name+'.pdb')
        lig_pdbqt_path = os.path.join(lig_pdbqt_save_dir, lig_name+'.pdbqt')
        rec_pdbqt_path = os.path.join(rec_pdbqt_save_dir, pdb_name+'.pdbqt')
        lig_ls.append(lig_pdbqt_path)
        rec_ls.append(rec_pdbqt_path)
        if not os.path.exists(lig_mol2_path):
            print('obabel -:\"{}\" -omol2 -O {} --gen3d'.format(smiles, lig_mol2_path))
            os.system('obabel -:\"{}\" -omol2 -O {} --gen3d'.format(smiles, lig_mol2_path))
        if not os.path.exists(lig_pdbqt_path):
            command = f'cd {lig_mol2_save_dir}; {pythonsh_path} {prepare_ligand4_path} -l {lig_mol2_path} -o {lig_pdbqt_path}'
            subprocess.run(command, shell=True, check=True)
        if not os.path.exists(rec_pdbqt_path):
            command = f'cd {rec_pdb_dir}; {pythonsh_path} {prepare_receptor4_path} -r {rec_pdb_path} -o {rec_pdbqt_path}'
            subprocess.run(command, shell=True, check=True)
    df['Ligand_PDBQT_Path'] = lig_ls
    df['Protein_PDBQT_Path'] = rec_ls



def batch_pdb2pdbqt(task_name):
    protein_pdbqt_save_dir = parent_dir+f'/task/{task_name}/protein/pdbqts'
    protein_pdb_dir = parent_dir+f'/task/{task_name}/protein/pdbs'
    ligand_pdbqt_save_dir = parent_dir+f'/task/{task_name}/ligand/pdbqts'
    ligand_mol2_save_dir = parent_dir+f'/task/{task_name}/ligand/mol2s'
    os.makedirs(protein_pdbqt_save_dir, exist_ok=True)
    os.makedirs(ligand_pdbqt_save_dir, exist_ok=True)
    os.makedirs(ligand_mol2_save_dir, exist_ok=True)
    df = pd.read_csv(f'task/{task_name}/{task_name}_POCKET.csv', sep='\t')
    generate_pdbqt(df, ligand_mol2_save_dir, ligand_pdbqt_save_dir, protein_pdb_dir, protein_pdbqt_save_dir)
    df.to_csv(f'task/{task_name}/{task_name}_POCKET.csv', sep='\t', index=False)


def extract_best_conformation(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    best_score = float('inf')
    best_model = []
    current_model = []
    recording = False
    for line in lines:
        if line.startswith("MODEL"):
            recording = True
            current_model = []
        elif "ENDMDL" in line:
            recording = False
            score_line = [l for l in current_model if "REMARK VINA RESULT" in l]
            if score_line:
                score = float(score_line[0].split()[3])
                if score < best_score:
                    best_score = score
                    best_model = current_model
        if recording:
            current_model.append(line)
    return best_model, best_score

def run_vina_and_extract_best(task_dir, row, ligand_pdbqt_path, receptor_pdbqt_path, repeat_times):

    log_dir = task_dir+'/log'
    output_dir = task_dir+'/output'
    best_pdbqt_dir = task_dir+'/best_pdbqt'
    best_mol2_dir = task_dir+'/best_mol2'
    for file in  [log_dir, output_dir, best_pdbqt_dir, best_mol2_dir]:
        os.makedirs(file, exist_ok=True)
    for dir in [task_dir, log_dir, output_dir, best_pdbqt_dir, best_mol2_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    # with open(config_file, 'r') as f:
    #     config_options = f.read().strip()

    best_scores = []
    for i in range(repeat_times):
        log_file = f"{log_dir}/run_{i + 1}.log"
        output_file = f"{output_dir}/run_{i + 1}.pdbqt"
        print(f"{VINA} --center_x {row['center_x']} --center_y {row['center_y']} --center_z {row['center_z']} --size_x {row['size_x']} --size_y {row['size_y']} --size_z {row['size_z']} --ligand {ligand_pdbqt_path} --receptor {receptor_pdbqt_path} --log {log_file} --out {output_file}")
        subprocess.run(f"{VINA} --center_x {row['center_x']} --center_y {row['center_y']} --center_z {row['center_z']} --size_x {row['size_x']} --size_y {row['size_y']} --size_z {row['size_z']} --ligand {ligand_pdbqt_path} --receptor {receptor_pdbqt_path} --log {log_file} --out {output_file}", shell=True)
        best_model, best_score = extract_best_conformation(output_file)
        best_scores.append(best_score)
        best_pdbqt_file = f"{best_pdbqt_dir}/run_{i + 1}_best.pdbqt"
        with open(best_pdbqt_file, 'w') as f:
            f.writelines(best_model)
        best_mol2_file = f"{best_mol2_dir}/run_{i + 1}_best.mol2"
        cmd = [OBABEL, '-ipdbqt', best_pdbqt_file, '-omol2', '-O', best_mol2_file]
        subprocess.run(cmd, check=True)

    with open(f"{task_dir}/best_scores.txt", 'w') as score_file:
        for index, score in enumerate(best_scores, 1):
            score_file.write(f"Run {index}: {score}\n")


def task_batch_dock(task_name):
    df = pd.read_csv(f'task/{task_name}/{task_name}_POCKET.csv', sep='\t')
    for index,row in df.iterrows():
        pdb_name = row['Protein_UniProtID']
        lig_name = row['Ligand_ID']
        os.makedirs(f'task/{task_name}/dock/{pdb_name}-{lig_name}', exist_ok=True)
        run_vina_and_extract_best(task_dir=f'task/{task_name}/dock/{pdb_name}-{lig_name}', 
                                row=row, 
                                ligand_pdbqt_path=row['Ligand_PDBQT_Path'], 
                                receptor_pdbqt_path=row['Protein_PDBQT_Path'],
                                repeat_times=20,)


def calculate_rmsd_simple(mol1, mol2):
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    coords1 = np.array([conf1.GetAtomPosition(i) for i in range(mol1.GetNumAtoms())])
    coords2 = np.array([conf2.GetAtomPosition(i) for i in range(mol2.GetNumAtoms())])
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def calculate_rmsd_matrix(conformers):
    num_conformers = len(conformers)
    rmsd_matrix = np.zeros((num_conformers, num_conformers))
    for i in range(num_conformers):
        for j in range(i + 1, num_conformers):
            rmsd = calculate_rmsd_simple(conformers[i],conformers[j])
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd
    return rmsd_matrix

def plot(rmsd_matrix_path, out_png_path):
    rmsd_mat = np.loadtxt(rmsd_matrix_path)
    Z = linkage(squareform(rmsd_mat), method='single')
    new_labels = [str(int(label) + 1) for label in range(rmsd_mat.shape[0])]
    plt.figure()
    dendrogram(Z, labels=new_labels)
    plt.title('Hierarchical Clustering of Conformers')
    plt.xlabel('Conformer index')
    plt.ylabel('RMSD (Å)')
    plt.tight_layout()
    plt.savefig(f'{os.path.dirname(rmsd_matrix_path)}/cluster.png', dpi=600)
    plt.close()


def calculate_rmsd_matrix_mol2(mol2_dir):
    conformers = [Chem.MolFromMol2File(f'{mol2_dir}/run_{i}_best.mol2', removeHs=False, sanitize=False) for i in range(1,21)]
    rmsd_matrix = calculate_rmsd_matrix(conformers)
    rmsd_matrix_path = f'{os.path.dirname(mol2_dir)}/rmsd_matrix.txt'
    rmsd_cluster_png_path = f'{os.path.dirname(mol2_dir)}/rmsd_cluster.png'
    np.savetxt(rmsd_matrix_path, rmsd_matrix)
    plot(rmsd_matrix_path, rmsd_cluster_png_path)
    return rmsd_matrix_path

def find_representative_by_mean_rmsd(rmsd_matrix_path):
    rmsd_mat = np.loadtxt(rmsd_matrix_path)
    mean_rmsd = np.mean(rmsd_mat, axis=1)
    representative_idx = np.argmin(mean_rmsd)
    shutil.copy(os.path.dirname(rmsd_matrix_path)+f'/best_mol2/run_{representative_idx+1}_best.mol2', 
                os.path.dirname(rmsd_matrix_path)+f'/center.mol2')
    return representative_idx



def cluster_and_find_representative(rmsd_matrix_path, cutoff=30):
    rmsd_mat = np.loadtxt(rmsd_matrix_path)
    Z = linkage(squareform(rmsd_mat), method='single')
    clusters = fcluster(Z, cutoff, criterion='distance') 
    unique_clusters = np.unique(clusters)
    max_cluster = max(unique_clusters, key=lambda x: np.sum(clusters == x))
    indices = np.where(clusters == max_cluster)[0]
    intra_cluster_rmsd = np.sum(rmsd_mat[indices][:, indices], axis=0)
    representative_idx = indices[np.argmin(intra_cluster_rmsd)]
    print(f"Cluster {max_cluster} is the largest with representative molecule {representative_idx+1}.")
    shutil.copy(os.path.dirname(rmsd_matrix_path)+f'/best_mol2/run_{representative_idx+1}_best.mol2', os.path.dirname(rmsd_matrix_path)+f'/run_{representative_idx+1}_best.mol2')
    return representative_idx


def cluster_and_find_representative_auto(rmsd_matrix_path):
    rmsd_mat = np.loadtxt(rmsd_matrix_path)
    Z = linkage(squareform(rmsd_mat), method='single')
    new_labels = [str(int(label) + 1) for label in range(rmsd_mat.shape[0])]
    dendro_info = dendrogram(Z, labels=new_labels)
    color_dict = {}
    for i, label in enumerate(dendro_info['ivl']):
        cluster_color = dendro_info['leaves_color_list'][i]
        color_dict[label] = cluster_color
    color_groups = {}
    for label, color in color_dict.items():
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(label)
    max_cluster_color = max(color_groups, key=lambda k: len(color_groups[k]))
    max_cluster = color_groups[max_cluster_color]
    indices = [int(i)-1 for i in max_cluster]
    intra_cluster_rmsd = np.sum(rmsd_mat[indices][:, indices], axis=0)
    representative_idx = indices[np.argmin(intra_cluster_rmsd)]
    print(f"Cluster {max_cluster} is the largest with representative molecule {representative_idx+1}.")
    shutil.copy(os.path.dirname(rmsd_matrix_path)+f'/best_mol2/run_{representative_idx+1}_best.mol2', os.path.dirname(rmsd_matrix_path)+f'/run_{representative_idx+1}_best.mol2')
    # os.system(f'rm '+ os.path.dirname(rmsd_matrix_path)+f'/center.mol2')
    return representative_idx, os.path.dirname(rmsd_matrix_path)+f'/run_{representative_idx+1}_best.mol2'


def task_batch_center(task_name):
    df = pd.read_csv(f'task/{task_name}/{task_name}_POCKET.csv', sep='\t')
    score_ls = []
    path_ls = []
    rmsd_ls = []
    for index,row in df.iterrows():
        code = row['Protein_UniProtID']
        lig = row['Ligand_ID']
        mol2_dir = f'task/{task_name}/dock/{code}-{lig}/best_mol2'
        scores_file = f'task/{task_name}/dock/{code}-{lig}/best_scores.txt'
        scores = []
        with open(scores_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                scores.append(float(line.split(':')[1].strip())) 
        rsmd_matrix_path = calculate_rmsd_matrix_mol2(mol2_dir)
        # rsmd_matrix_path = f'task/{task_name}/dock/{code}/rmsd_matrix.txt'
        representative_idx, representative_path = cluster_and_find_representative_auto(rsmd_matrix_path) 
        score_ls.append(scores[representative_idx])  
        path_ls.append(representative_path)  
        mol1 = Chem.MolFromMol2File(representative_path, removeHs=False, sanitize=False) 
        OBABEL = 'obabel'
        cry_pdbqt = f"task/{task_name}/ligand/pdbqts/{lig}.pdbqt"
        cry_path = f'task/{task_name}/ligand/mol2s/{lig}.mol2'
        os.system(f'{OBABEL} -ipdbqt {cry_pdbqt} -omol2 -O {cry_path}')
        mol2 = Chem.MolFromMol2File(cry_path, removeHs=False, sanitize=False)
        rmsd = calculate_rmsd_simple(mol1, mol2)
        rmsd_ls.append(rmsd)
    df['representative affinity'] = score_ls
    df['representative mol2'] = path_ls
    df['dock RMSD'] = rmsd_ls
    df.to_csv(f'task/{task_name}/{task_name}_Result.csv', sep='\t', index=False)




# batch_pdb2pdbqt('5SQQ')
# task_batch_dock('5SQQ')
task_batch_center('5SQQ')



