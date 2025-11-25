from model import *

class ProtGraphTransformerWoEdge(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.h = hidden_dim
        self.nh = num_heads
        self.d = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.dropout_h = nn.Dropout(dropout)
        self.mlp_node = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.residual = residual

    def forward(self, g, h):
        g.ndata['h'] = h
        Q = self.q(h).reshape(-1, self.nh, self.d)
        K = self.k(h).reshape(-1, self.nh, self.d)
        V = self.v(h).reshape(-1, self.nh, self.d)
        g.ndata['Q'] = Q
        g.ndata['K'] = K
        g.ndata['V'] = V
        g.apply_edges(lambda E: {
            's': (E.dst['Q'] * E.src['K']).sum(-1) / (self.d ** 0.5),
            'V': E.src['V']    
        })
        g.edata['a'] = edge_softmax(g, g.edata['s'])
        g.edata['msg'] = g.edata['a'].unsqueeze(-1) * g.edata['V']
        g.update_all(fn.copy_e('msg', 'msg'), fn.sum('msg', 'h_new'))
        h_new = g.ndata.pop('h_new').reshape(-1, self.nh * self.d)
        h_new = self.mlp_node(h_new)
        if self.residual:
            h = self.norm_h(h + self.dropout_h(h_new))
        else:
            h = self.norm_h(self.dropout_h(h_new))
        return h


class Cov2ProtGraphTransformerWoEdge(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.layer1 = ProtGraphTransformerWoEdge(hidden_dim, num_heads, dropout, residual)
        self.layer2 = ProtGraphTransformerWoEdge(hidden_dim, num_heads, dropout, residual)
    def forward(self, g, h, ):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        return h



class PLCA_Pocket_Ablation(nn.Module):
    def __init__(self, in_dim_dict={'prot_ankh':1536, 'prot_esm2':320, 'prot_phychem':42, 'prot_node_local':147, 'prot_edge_local':73, 'ligand':40}, 
                 max_protein_length=1500, max_ligand_length=225, dropout=0.1, hidden_dim=256, num_layers=2, ablation_name='wo_ankh'):
        super().__init__()
        self.num_layers = num_layers
        self.lig_mlp = MLP(in_dim_dict['ligand'], hidden_dim, hidden_dim)
        self.prot_llm1_mlp = MLP(in_dim_dict['prot_ankh'], hidden_dim, hidden_dim)
        self.prot_llm2_mlp = MLP(in_dim_dict['prot_esm2'], hidden_dim, hidden_dim)
        self.prot_phychem_mlp = MLP(in_dim_dict['prot_phychem'], hidden_dim, hidden_dim)
        self.prot_local_mlp = MLP(in_dim_dict['prot_node_local'], hidden_dim, hidden_dim)
        self.prot_node_mlp = MLP(hidden_dim*3, hidden_dim*2, hidden_dim)
        self.prot_edge_mlp = MLP(in_dim_dict['prot_edge_local'], hidden_dim, hidden_dim)
        self.lig_graph_encoders = nn.ModuleList([Cov2MolGraphTransformer(hidden_dim,num_heads=4) for _ in range(num_layers)]) 
        self.lig_pos_encoder = PositionalEncoding(hidden_dim, max_len=max_ligand_length)
        if ablation_name == 'wo_geometric':
            self.prot_graph_encoders=nn.ModuleList([Cov2ProtGraphTransformerWoEdge(hidden_dim,num_heads=4) for _ in range(num_layers)])
        else:
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
        self.ablation_name = ablation_name

    def forward(self, prot_graph, esm2_padded, lig_graph, mask_prot, mask_lig):
        lig_g_node = self.lig_mlp(lig_graph.ndata['atom_feat'])
        prot_llm1 = self.prot_llm1_mlp(prot_graph.ndata['ankh'])        
        prot_llm2 = self.prot_llm2_mlp(prot_graph.ndata['esm2'])
        prot_phychem = self.prot_phychem_mlp(prot_graph.ndata['phychem'])
        prot_node_local = self.prot_local_mlp(prot_graph.ndata['local'])
        if self.ablation_name == 'wo_ankh':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm2, prot_phychem, prot_node_local], dim=-1))
        elif self.ablation_name == 'wo_phychem':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2, prot_node_local], dim=-1))
        elif self.ablation_name == 'wo_geometric':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2, prot_phychem], dim=-1))
        elif self.ablation_name == 'wo_esm2':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_phychem, prot_node_local], dim=-1))       
        prot_g_edge = self.prot_edge_mlp(prot_graph.edata['local'])
        for i in range(self.num_layers):
            lig_g_node = self.lig_graph_encoders[i](lig_graph, lig_g_node)
            lig_node = pad_from_mask(lig_g_node, mask_lig)
            if i ==0: 
                lig_node = self.lig_pos_encoder(lig_node) 
            if self.ablation_name != 'wo_geometric':
                prot_g_node, prot_g_edge = self.prot_graph_encoders[i](prot_graph, prot_g_node, prot_g_edge)  
            else:
                prot_g_node = self.prot_graph_encoders[i](prot_graph, prot_g_node)  
            prot_node = pad_from_mask(prot_g_node, mask_prot) 
            prot_node = self.prot2lig_CA[i](q=prot_node,k=lig_node,v=lig_node,k_mask=mask_lig)*mask_prot.unsqueeze(-1).float()
            lig_node = self.lig2prot_CA[i](q=lig_node,k=prot_node,v=prot_node,k_mask=mask_prot)*mask_lig.unsqueeze(-1).float()
            prot_g_node = unpad_from_mask(prot_node, mask_prot)
            lig_g_node = unpad_from_mask(lig_node, mask_lig)
        pocket_logits = self.pocket_predictor(prot_node)
        return pocket_logits




class PLCA_Affinity_Ablation(nn.Module):
    def __init__(self, in_dim_dict={'prot_ankh':1536, 'prot_esm2':320, 'prot_phychem':42, 'prot_node_local':147, 'prot_edge_local':73, 'ligand':40}, 
                 max_protein_length=1500, max_ligand_length=225, dropout=0.1, hidden_dim=256, num_layers=2, ablation_name='raw', num_pred_heads=1, use_assay=False):
        super().__init__()
        self.num_layers = num_layers
        self.lig_mlp = MLP(in_dim_dict['ligand'], hidden_dim, hidden_dim)
        self.prot_llm1_mlp = MLP(in_dim_dict['prot_ankh'], hidden_dim, hidden_dim)
        self.prot_llm2_mlp = MLP(in_dim_dict['prot_esm2'], hidden_dim, hidden_dim)
        self.prot_phychem_mlp = MLP(in_dim_dict['prot_phychem'], hidden_dim, hidden_dim)
        self.prot_local_mlp = MLP(in_dim_dict['prot_node_local'], hidden_dim, hidden_dim)
        if ablation_name == 'w_pocket':
            self.prot_node_mlp = MLP(hidden_dim*4, hidden_dim*2, hidden_dim)
        else:
            self.prot_node_mlp = MLP(hidden_dim*3, hidden_dim*2, hidden_dim)
        self.prot_edge_mlp = MLP(in_dim_dict['prot_edge_local'], hidden_dim, hidden_dim)
        self.lig_graph_encoders = nn.ModuleList([Cov2MolGraphTransformer(hidden_dim,num_heads=4) for _ in range(num_layers)]) 
        self.lig_pos_encoder = PositionalEncoding(hidden_dim, max_len=max_ligand_length)
        if ablation_name == 'wo_geometric':
            self.prot_graph_encoders=nn.ModuleList([Cov2ProtGraphTransformerWoEdge(hidden_dim,num_heads=4) for _ in range(num_layers)])
        else:
            self.prot_graph_encoders=nn.ModuleList([Cov2ProtGraphTransformer(hidden_dim,num_heads=4) for _ in range(num_layers)])
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
        self.max_pocket_length = 55
        self.ablation_name = ablation_name

    def forward(self, prot_graph, esm2_padded, lig_graph, assay_feat, mask_prot, mask_lig, mask_assay, pocket_label_padded):
        lig_g_node = self.lig_mlp(lig_graph.ndata['atom_feat'])
        prot_llm1 = self.prot_llm1_mlp(prot_graph.ndata['ankh'])        
        prot_llm2 = self.prot_llm2_mlp(prot_graph.ndata['esm2'])
        prot_phychem = self.prot_phychem_mlp(prot_graph.ndata['phychem'])
        prot_node_local = self.prot_local_mlp(prot_graph.ndata['local'])
        if self.ablation_name == 'wo_ankh':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm2, prot_phychem, prot_node_local], dim=-1))
        elif self.ablation_name == 'wo_phychem':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2, prot_node_local], dim=-1))
        elif self.ablation_name == 'wo_geometric':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2, prot_phychem], dim=-1))
        elif self.ablation_name == 'wo_esm2':
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_phychem, prot_node_local], dim=-1))    
        else:
            prot_g_node = self.prot_node_mlp(torch.concat([prot_llm1, prot_llm2, prot_phychem, prot_node_local], dim=-1))    
        prot_g_edge = self.prot_edge_mlp(prot_graph.edata['local'])
        for i in range(self.num_layers):
            feat_assay = self.assay_1d_encoders[i](assay_feat) * mask_assay.unsqueeze(-1).float() if self.use_assay else None # [batch,len,dim]
            lig_g_node = self.lig_graph_encoders[i](lig_graph, lig_g_node)
            lig_node = pad_from_mask(lig_g_node, mask_lig)
            if i ==0: 
                lig_node = self.lig_pos_encoder(lig_node) 
            if self.ablation_name == 'wo_geometric':
                prot_g_node = self.prot_graph_encoders[i](prot_graph, prot_g_node) 
            else:
                prot_g_node, prot_g_edge = self.prot_graph_encoders[i](prot_graph, prot_g_node, prot_g_edge)  
            prot_node = pad_from_mask(prot_g_node, mask_prot) 
            prot_node = self.prot2lig_CA[i](q=prot_node,k=lig_node,v=lig_node,k_mask=mask_lig)*mask_prot.unsqueeze(-1).float()
            lig_node = self.lig2prot_CA[i](q=lig_node,k=prot_node,v=prot_node,k_mask=mask_prot)*mask_lig.unsqueeze(-1).float()
            prot_g_node = unpad_from_mask(prot_node, mask_prot)
            lig_g_node = unpad_from_mask(prot_node, mask_lig)
        feat_prot = prot_node.mean(dim=1) # [batch,dim]
        feat_lig = lig_node.mean(dim=1)
        feat_assay = feat_assay.mean(dim=1) if self.use_assay else None
        feat_total = torch.cat([feat_prot, feat_lig, feat_assay], dim=-1) if self.use_assay else torch.cat([feat_prot, feat_lig], dim=-1) # [batch,dim*3] or #[batch,dim*2]                               
        feat_total = self.shared_fc(feat_total)
        affinitys = []
        for i, affinity_predictor in enumerate(self.affinity_headers):
            affinitys.append(affinity_predictor(feat_total))
        return torch.cat(affinitys, dim=1) # [batch,num_heads]