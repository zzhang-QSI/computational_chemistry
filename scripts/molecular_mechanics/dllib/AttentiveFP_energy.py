import itertools

import dgl
import mdtraj
import torch
import torch.nn as nn

from dgl.model_zoo.chem import AttentiveFP
from dgl import graph
from dgl.heterograph import  AdaptedHeteroGraph
from dgl.data.chem import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from dgl.data.chem.utils.featurizers import one_hot_encoding

class AttentiveFP_energy(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
     Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

     Parameters
     ----------
     node_feat_size : int
         Size for the input node (atom) features.
     edge_feat_size : int
         Size for the input edge (bond) features.
     num_layers : int
         Number of GNN layers.
     num_timesteps : int
         Number of timesteps for updating the molecular representation with GRU.
     graph_feat_size : int
         Size of the learned graph representation (molecular fingerprint).
     output_size : int
         Size of the prediction (target labels).
     dropout : float
         The probability for performing dropout.
     """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 num_timesteps,
                 graph_feat_size,
                 output_size,
                 dropout):
        super(AttentiveFP_energy, self).__init__()
        self.base_model=AttentiveFP(node_feat_size,
                 edge_feat_size,
                 num_layers,
                 num_timesteps,
                 graph_feat_size,
                 output_size,
                 dropout)

    def forward(self, protein_graph,split_idx=None):
        """Apply the model for prediction.
        Parameters
        ----------
        graph : DGLHeteroGraph
            DGLHeteroGraph consisting of the ligand graph, the protein graph
            and the complex graph, along with preprocessed features.
        Returns
        -------
        Float32 tensor of shape (B, O)
            Predicted protein-ligand binding affinity. B for the number
            of protein-ligand pairs in the batch and O for the number of tasks.
        """



        protein_graph_node_feats = protein_graph.ndata['h']

        protein_graph_distances = protein_graph.edata['e']


        pred_energy =self.base_model(protein_graph,
                                             protein_graph_node_feats,
                                             protein_graph_distances)


        return pred_energy

from dllib.xyz2mol import mmmol2rdkitmol
def hetero2homo_graph(hetero_graph,mol):
    rd_mol = mmmol2rdkitmol(mol)

    atom_featurizer = CanonicalAtomFeaturizer()
    ### convert DGL hetero graph to DGL graph
    homo_graph = dgl.DGLGraph()
    homo_graph.from_networkx(hetero_graph.to_networkx())
    homo_graph.ndata.update(atom_featurizer(rd_mol))
    homo_graph.edata['e'] = hetero_graph.edata['distance']
    return homo_graph


def collate(data):

    indices,  protein_mols, graphs, labels = map(list, zip(*data))
    for i in range(len(protein_mols)):
        # rd_mol=mmmol2rdkitmol(protein_mols[i])
        # ### convert DGL hetero graph to DGL graph
        # homo_graph=dgl.DGLGraph()
        # homo_graph.from_networkx( graphs[i].to_networkx())
        # homo_graph.ndata.update(atom_featurizer(rd_mol))
        # homo_graph.edata['e']=graphs[i].edata['distance']
        graphs[i]=hetero2homo_graph(graphs[i],protein_mols[i])



    bg = dgl.batch(graphs)

    labels = torch.stack(labels, dim=0)

    return indices,   protein_mols, bg, labels
if __name__ == '__main__':
    import glob
    from mmlib.molecule import Molecule
    from dllib.mmlib_utils import get_mol_3D_coordinates
    from dllib.xyz2graph import  XYZDataSet

    from dllib.ACNN_energy import  k_nearest_neighbors_torch
    from torch.utils.data import DataLoader
    from torch.optim import Adam


    dataset = XYZDataSet(glob.glob("../../../geom/xyzq/*.xyzq"),zero_padding=False)


    #######add features to edge and node ##########

    energy_model=AttentiveFP_energy( node_feat_size=74,
                 edge_feat_size=1,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 output_size=8,
                 dropout=0.2 )
    optimizer=Adam(energy_model.parameters(), lr=10 ** (-2.5))
    i=0
    ###train a energy function
    while i<10:
        i+=1
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=6,
                                  shuffle=True,
                                  collate_fn=collate)
        total_loss=0

        for i_batch, sample_batched in enumerate(train_loader):
            # train_graphs=sample_batched[2][('protein_atom', 'protein', 'protein_atom')]
            # train_graphs.batch_size=sample_batched[2].batch_size
            loss=torch.nn.functional.mse_loss(energy_model(sample_batched[2]) , sample_batched[3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=(float(loss))
        print(total_loss)

    refine_graph=dataset[0][2]

    test_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=collate)


    for i_batch, sample_batched in enumerate(test_loader):
        ## get coodinates and parse as torch tensor
        coords = torch.FloatTensor(get_mol_3D_coordinates(sample_batched[1][0]))
        coords=torch.nn.Parameter(coords)
        num_protein_atoms=coords.shape[0]
        optimizer = Adam([coords], lr=1e-4)
        for optimize_step in range(1000):
            ## compute edge distance
            protein_srcs,protein_dsts,edge_distance=k_nearest_neighbors_torch(coords, neighbor_cutoff=12.,
                                              max_num_neighbors=12,)


            sample_batched[2].edata['e'] = edge_distance.reshape(-1,1)
            ## optim

            total_energy=energy_model(sample_batched[2])[0][0]
            optimizer.zero_grad()
            total_energy.backward()
            optimizer.step()
            print("predict energy",total_energy,coords[0])
        break