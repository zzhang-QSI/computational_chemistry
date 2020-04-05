"""Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity"""
# pylint: disable=C0103, C0123
import itertools

import mdtraj
import torch
import torch.nn as nn

from dgl.nn.pytorch import AtomicConv


def truncated_normal_(tensor, mean=0., std=1.):
    """Fills the given tensor in-place with elements sampled from the truncated normal
    distribution parameterized by mean and std.
    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than 2 std
    from the mean are dropped.
    We credit to Ruotian Luo for this implementation:
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15.
    Parameters
    ----------
    tensor : Float32 tensor of arbitrary shape
        Tensor to be filled.
    mean : float
        Mean of the truncated normal distribution.
    std : float
        Standard deviation of the truncated normal distribution.
    """
    shape = tensor.shape
    tmp = tensor.new_empty(shape + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class ACNNPredictor(nn.Module):
    """Predictor for ACNN.
    Parameters
    ----------
    in_size : int
        Number of radial filters used.
    hidden_sizes : list of int
        Specifying the hidden sizes for all layers in the predictor.
    weight_init_stddevs : list of float
        Specifying the standard deviations to use for truncated normal
        distributions in initialzing weights for the predictor.
    dropouts : list of float
        Specifying the dropouts to use for all layers in the predictor.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.
    num_tasks : int
        Output size.
    """
    def __init__(self, in_size, hidden_sizes, weight_init_stddevs,
                 dropouts, features_to_use, num_tasks):
        super(ACNNPredictor, self).__init__()

        if type(features_to_use) != type(None):
            in_size *= len(features_to_use)

        modules = []
        for i, h in enumerate(hidden_sizes):
            linear_layer = nn.Linear(in_size, h)
            truncated_normal_(linear_layer.weight, std=weight_init_stddevs[i])
            modules.append(linear_layer)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropouts[i]))
            in_size = h
        linear_layer = nn.Linear(in_size, num_tasks)
        truncated_normal_(linear_layer.weight, std=weight_init_stddevs[-1])
        modules.append(linear_layer)
        self.project = nn.Sequential(*modules)
        self.num_tasks=num_tasks

    def forward(self, batch_size,
                protein_conv_out,split_idx=None):
        """Perform the prediction.
        Parameters
        ----------
        batch_size : int
            Number of datapoints in a batch.
        frag1_node_indices_in_complex : Int64 tensor of shape (V1)
            Indices for atoms in the first fragment (protein) in the batched complex.
        frag2_node_indices_in_complex : list of int of length V2
            Indices for atoms in the second fragment (ligand) in the batched complex.
        ligand_conv_out : Float32 tensor of shape (V2, K * T)
            Updated ligand node representations. V2 for the number of atoms in the
            ligand, K for the number of radial filters, and T for the number of types
            of atomic numbers.
        protein_conv_out : Float32 tensor of shape (V1, K * T)
            Updated protein node representations. V1 for the number of
            atoms in the protein, K for the number of radial filters,
            and T for the number of types of atomic numbers.
        complex_conv_out : Float32 tensor of shape (V1 + V2, K * T)
            Updated complex node representations. V1 and V2 separately
            for the number of atoms in the ligand and protein, K for
            the number of radial filters, and T for the number of
            types of atomic numbers.
        Returns
        -------
        Float32 tensor of shape (B, O)
            Predicted protein-ligand binding affinity. B for the number
            of protein-ligand pairs in the batch and O for the number of tasks.
        """

        protein_feats = self.project(protein_conv_out) # (V2, O)

        if split_idx is None:
            protein_energy = protein_feats.reshape( batch_size,-1,self.num_tasks).sum(1) # (B, O)
        else:
            frag1_node_indices_in_complex =torch.arange(split_idx).to(protein_feats.device)
            frag2_node_indices_in_complex = torch.arange(split_idx,protein_feats.shape[0]).to(protein_feats.device)

            complex_ligand_energy = protein_feats[frag1_node_indices_in_complex].reshape(
                 batch_size, -1).sum(-1, keepdim=True)
            complex_protein_energy = protein_feats[frag2_node_indices_in_complex].reshape(
                 batch_size, -1).sum(-1, keepdim=True)
            protein_energy = complex_ligand_energy + complex_protein_energy

        return   protein_energy

class ACNN_energy(nn.Module):
    """Atomic Convolutional Networks.
    The model was proposed in `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.
    Parameters
    ----------
    hidden_sizes : list of int
        Specifying the hidden sizes for all layers in the predictor.
    weight_init_stddevs : list of float
        Specifying the standard deviations to use for truncated normal
        distributions in initialzing weights for the predictor.
    dropouts : list of float
        Specifying the dropouts to use for all layers in the predictor.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.
    radial : None or list
        If not None, the list consists of 3 lists of floats, separately for the
        options of interaction cutoff, the options of rbf kernel mean and the
        options of rbf kernel scaling. If None, a default option of
        ``[[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]`` will be used.
    num_tasks : int
        Number of output tasks.
    """

    def __init__(self, hidden_sizes, weight_init_stddevs, dropouts,
                 features_to_use=None, radial=None, num_tasks=1):
        super(ACNN_energy, self).__init__()

        if radial is None:
            radial = [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]
        # Take the product of sets of options and get a list of 3-tuples.
        radial_params = [x for x in itertools.product(*radial)]
        radial_params = torch.stack(list(map(torch.tensor, zip(*radial_params))), dim=1)

        interaction_cutoffs = radial_params[:, 0]
        rbf_kernel_means = radial_params[:, 1]
        rbf_kernel_scaling = radial_params[:, 2]


        self.protein_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                       rbf_kernel_scaling, features_to_use)

        self.predictor = ACNNPredictor(radial_params.shape[0], hidden_sizes,
                                       weight_init_stddevs, dropouts, features_to_use, num_tasks)

    def forward(self, protein_graph):
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



        protein_graph_node_feats = protein_graph.ndata['atomic_number']
        assert protein_graph_node_feats.shape[-1] == 1
        protein_graph_distances = protein_graph.edata['distance']
        protein_conv_out = self.protein_conv(protein_graph,
                                             protein_graph_node_feats,
                                             protein_graph_distances)

        if "complex" in "".join( protein_graph.canonical_etypes[0]):
            split_idx=max( torch.where(protein_graph.ndata['_TYPE'] == 0)[0])+1
            pred_energy = self.predictor(protein_graph.batch_size,
                                         protein_conv_out,split_idx=split_idx)
        else:
            pred_energy= self.predictor(protein_graph.batch_size,
             protein_conv_out)

        return pred_energy



def k_nearest_neighbors_torch(coordinates_torch:torch.FloatTensor, neighbor_cutoff, max_num_neighbors):
    """Find k nearest neighbors for each atom based on the 3D coordinates.

    Parameters
    ----------
    coordinates : numpy.ndarray of shape (N, 3)
        The 3D coordinates of atoms in the molecule. N for the number of atoms.
    neighbor_cutoff : float
        Distance cutoff to define 'neighboring'.
    max_num_neighbors : int or None.
        If not None, then this specifies the maximum number of closest neighbors
        allowed for each atom.

    Returns
    -------
    neighbor_list : dict(int -> list of ints)
        Mapping atom indices to their k nearest neighbors.
    """
    coordinates=coordinates_torch.data.numpy()
    num_atoms = coordinates.shape[0]
    traj = mdtraj.Trajectory(coordinates.reshape((1, num_atoms, 3)), None)
    # TODO: may consider to force the covalent bond has maximum distance
    neighbors = mdtraj.geometry.compute_neighborlist(traj, neighbor_cutoff)
    srcs, dsts, distances = [], [], []
    for i in range(num_atoms):
        delta = coordinates_torch[i] - coordinates_torch[ neighbors[i]]
        dist = torch.norm(delta, 2,dim=1)
        if max_num_neighbors is not None and len(neighbors[i]) > max_num_neighbors:
            sorted_neighbors = list(zip(dist, neighbors[i]))
            # Sort neighbors based on distance from smallest to largest
            sorted_neighbors.sort(key=lambda tup: tup[0])
            dsts.extend([i for _ in range(max_num_neighbors)])
            srcs.extend([int(sorted_neighbors[j][1]) for j in range(max_num_neighbors)])
            distances.extend([ sorted_neighbors[j][0].view(1,-1)  for j in range(max_num_neighbors)])
        else:
            dsts.extend([i for _ in range(len(neighbors[i]))])
            srcs.extend(neighbors[i].tolist())
            distances.extend(dist)

    return srcs, dsts, torch.cat(distances)


if __name__ == '__main__':
    import glob
    from mmlib.molecule import Molecule
    from dllib.mmlib_utils import get_mol_3D_coordinates
    from dllib.xyz2graph import  XYZDataSet,collate
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from dgl import graph

    dataset = XYZDataSet(glob.glob("../../../geom/xyzq/*.xyzq"))




    energy_model=ACNN_energy([128, 128, 64], [0.125, 0.125, 0.177, 0.01], [0. , 0. , 0.], torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]), num_tasks=8)
    optimizer=Adam(energy_model.parameters(), lr=1e-4)
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
        for optimize_step in range(100):
            ## compute edge distance
            protein_srcs,protein_dsts,edge_distance=k_nearest_neighbors_torch(coords, neighbor_cutoff=12.,
                                              max_num_neighbors=12,)

            protein_graph = graph((protein_srcs, protein_dsts),
                                  'protein_atom', 'protein', num_protein_atoms)

            ## optim
            protein_graph.edata['distance']=edge_distance.reshape(-1,1)
            protein_graph.nodes['protein_atom'].data['atomic_number']=sample_batched[2].nodes['protein_atom'].data['atomic_number'][:num_protein_atoms]
            protein_graph.nodes['protein_atom'].data['mask']= sample_batched[2].nodes['protein_atom'].data['mask'][:num_protein_atoms]
            protein_graph.batch_size=1
            total_energy=energy_model(protein_graph)[0][0]
            optimizer.zero_grad()
            total_energy.backward()
            optimizer.step()
            print("predict energy",total_energy,coords[0])
