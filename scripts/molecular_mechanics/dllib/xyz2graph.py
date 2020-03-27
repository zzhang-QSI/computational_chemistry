"""
Zhizhuo Zhang
03/23/2020
read XYZ format , and generate DGL graph
"""
from mmlib.molecule import Molecule




import dgl
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from dgl.data.chem.utils import k_nearest_neighbors
from dgl import graph
from dllib.mmlib_utils  import multiprocess_load_molecules
from dgl import backend as F


def filter_out_hydrogens(mol):
    """Get indices for non-hydrogen atoms.

    Parameters
    ----------
    mol : mmlib.molecule.Molecule
         instance.

    Returns
    -------
    indices_left : list of int
        Indices of non-hydrogen atoms.
    """
    indices_left = []
    for i, atom in enumerate(mol.atoms):
        atomic_num = int(atom.mass)
        # Hydrogen atoms have an atomic number of 1.
        if atomic_num  != 1:
            indices_left.append(i)
    return indices_left

def get_atomic_numbers(mol, indices):
    """Get the atomic numbers for the specified atoms.

    Parameters
    ----------
    mol :mmlib.molecule.Molecule
          molecule instance.
    indices : list of int
        Specifying atoms.

    Returns
    -------
    list of int
        Atomic numbers computed.
    """
    atomic_numbers = []
    for i in indices:
        atom = mol.atoms[i]
        atomic_numbers.append(int(atom.mass))
    return atomic_numbers

def XYZ_graph_construction_and_featurization( protein_mol,
                                              protein_coordinates,
                                              max_num_protein_atoms=None,
                                              neighbor_cutoff=12.,
                                              max_num_neighbors=12,
                                              strip_hydrogens=False):
    """Graph construction and featurization for `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    Parameters
    ----------

    protein_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    protein_coordinates : Float Tensor of shape (V2, 3)
        Atom coordinates in a protein.
    max_num_protein_atoms : int or None
        Maximum number of atoms in proteins for zero padding.
        If None, no zero padding will be performed. Default to None.
    neighbor_cutoff : float
        Distance cutoff to define 'neighboring'. Default to 12.
    max_num_neighbors : int
        Maximum number of neighbors allowed for each atom. Default to 12.
    strip_hydrogens : bool
        Whether to exclude hydrogen atoms. Default to False.
    """
    assert protein_coordinates is not None, 'Expect protein_coordinates to be provided.'

    if strip_hydrogens:
        # Remove hydrogen atoms and their corresponding coordinates
        protein_atom_indices_left = filter_out_hydrogens(protein_mol)
        protein_coordinates = protein_coordinates.take(protein_atom_indices_left, axis=0)
    else:
        protein_atom_indices_left = list(range(protein_mol.n_atoms ))

    # Compute number of nodes for each type


    if max_num_protein_atoms is None:
        num_protein_atoms = len(protein_atom_indices_left)
    else:
        num_protein_atoms = max_num_protein_atoms

    # Construct graph for atoms in the protein
    protein_srcs, protein_dsts, protein_dists = k_nearest_neighbors(
        protein_coordinates, neighbor_cutoff, max_num_neighbors)
    protein_graph = graph((protein_srcs, protein_dsts),
                          'protein_atom', 'protein', num_protein_atoms)
    protein_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        np.array(protein_dists).astype(np.float32)), (-1, 1))

    # Construct 4 graphs for complex representation, including the connection within
    # protein atoms, the connection within ligand atoms and the connection between
    # protein and ligand atoms.


    # Merge the graphs
    g = protein_graph
    protein_atomic_numbers = np.array(get_atomic_numbers(protein_mol, protein_atom_indices_left))
    # zero padding
    protein_atomic_numbers = np.concatenate([
        protein_atomic_numbers, np.zeros(num_protein_atoms - len(protein_atom_indices_left))])


    g.nodes['protein_atom'].data['atomic_number'] = F.reshape(F.zerocopy_from_numpy(
        protein_atomic_numbers.astype(np.float32)), (-1, 1))

    # Prepare mask indicating the existence of nodes

    protein_masks = np.zeros((num_protein_atoms, 1))
    protein_masks[:len(protein_atom_indices_left), :] = 1
    g.nodes['protein_atom'].data['mask'] = F.zerocopy_from_numpy(
        protein_masks.astype(np.float32))

    return g

class XYZDataSet(object):
    """Convert XYZ files to DGL graphs

    Parameters
    ----------
    add_hydrogens : bool
        Whether to add hydrogens via pdbfixer. Default to False.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``add_hydrogens`` and ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.
    construct_graph_and_featurize : callable
        Construct a DGLHeteroGraph for the use of GNNs. Mapping
        self.protein_mols[i],   and self.protein_coordinates[i]
        to a DGLHeteroGraph. Default to :func:`ACNN_graph_construction_and_featurization`.
    zero_padding : bool
        Whether to perform zero padding. While DGL does not necessarily require zero padding,
        pooling operations for variable length inputs can introduce stochastic behaviour, which
        is not desired for sensitive scenarios. Default to True.
    num_processes : int or None
        Number of worker processes to use. If None,
        then we will use the number of CPUs in the system. Default to 64.
    """

    def __init__(self,  xyz_files,  use_conformation=True,
                 construct_graph_and_featurize=XYZ_graph_construction_and_featurization,
                 zero_padding=True, num_processes=4):
        self.task_names = ['-logKd/Ki']
        self.n_tasks = len(self.task_names)


        self._preprocess(xyz_files,
                        use_conformation,
                         construct_graph_and_featurize, zero_padding, num_processes)

    def _filter_out_invalid(self, molecues_loaded, use_conformation):
        """Filter out invalid ligand-protein pairs.
        Parameters
        ----------
        molecues_loaded : list
            Each element is a 2-tuple of the RDKit molecule instance and its associated atom
            coordinates. None is used to represent invalid/non-existing molecule or coordinates.
        use_conformation : bool
            Whether we need conformation information (atom coordinates) and filter out molecules
            without valid conformation.
        """
        num_mols = len(molecues_loaded)
        self.indices,   self.protein_mols = [], []
        if use_conformation:
            self.protein_coordinates = []
        else:
            # Use None for placeholders.
            self.protein_coordinates = [None for _ in range(num_mols)]

        for i in range(num_mols):

            protein_mol, protein_coordinates = molecues_loaded[i]
            if (not use_conformation) and protein_mol is not None  :
                self.indices.append(i)
                self.protein_mols.append(protein_mol)
            elif all(v is not None for v in [
                protein_mol, protein_coordinates  ]):
                self.indices.append(i)
                self.protein_mols.append(protein_mol)
                self.protein_coordinates.append(protein_coordinates)

    def _preprocess(self,xyz_files,
                     use_conformation,
                    construct_graph_and_featurize, zero_padding, num_processes):
        """Preprocess the dataset.
        The pre-processing proceeds as follows:
        1. Load the dataset
        2. Clean the dataset and filter out invalid pairs
        3. Construct graphs
        4. Prepare node and edge features
        Parameters
        ----------
        xyz_files : str
             path for molecule files.

        use_conformation : bool
            Whether we need to extract molecular conformation from proteins and ligands.
        construct_graph_and_featurize : callable
            Construct a DGLHeteroGraph for the use of GNNs. Mapping self.ligand_mols[i],
            self.protein_mols[i], self.ligand_coordinates[i] and self.protein_coordinates[i]
            to a DGLHeteroGraph. Default to :func:`ACNN_graph_construction_and_featurization`.
        zero_padding : bool
            Whether to perform zero padding. While DGL does not necessarily require zero padding,
            pooling operations for variable length inputs can introduce stochastic behaviour, which
            is not desired for sensitive scenarios.
        num_processes : int or None
            Number of worker processes to use. If None,
            then we will use the number of CPUs in the system.
        """

        pdbs = [os.path.basename(x).split(".")[0] for x in xyz_files]
        self.protein_files=xyz_files

        num_processes = min(num_processes, len(pdbs))

        print('Loading molecues...')

        proteins_loaded = multiprocess_load_molecules(self.protein_files,
                                                      use_conformation=use_conformation,
                                                      num_processes=num_processes)

        self._filter_out_invalid(  proteins_loaded, use_conformation)



        self.labels = torch.stack([ torch.FloatTensor([mol_x.e_total,mol_x.e_kinetic,mol_x.e_potential,mol_x.e_nonbonded,mol_x.e_bonded,
                                  mol_x.e_bound,mol_x.e_elst,mol_x.e_vdw]) for mol_x,mol_coord in proteins_loaded])
        print('Finished cleaning the dataset, '
              'got {:d}/{:d} valid pairs'.format(len(self), len(pdbs)))

        # Prepare zero padding
        if zero_padding:

            max_num_protein_atoms = 0
            for i in range(len(self)):

                max_num_protein_atoms = max(
                    max_num_protein_atoms, self.protein_mols[i].n_atoms)
        else:

            max_num_protein_atoms = None

        print('Start constructing graphs and featurizing them.')

        self.graphs = []
        for i in range(len(self)):
            print('Constructing and featurizing datapoint {:d}/{:d}'.format(i+1, len(self)))
            self.graphs.append(construct_graph_and_featurize(
                 self.protein_mols[i],
                  self.protein_coordinates[i],
                  max_num_protein_atoms))

    def __len__(self):
        """Get the size of the dataset.
        Returns
        -------
        int
            Number of valid ligand-protein pairs in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, item):
        """Get the datapoint associated with the index.
        Parameters
        ----------
        item : int
            Index for the datapoint.
        Returns
        -------
        int
            Index for the datapoint.
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the ligand molecule.
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the protein molecule.
        DGLHeteroGraph
            Pre-processed DGLHeteroGraph with features extracted.
        Float32 tensor
            Label for the datapoint.
        """
        return item,  self.protein_mols[item], \
               self.graphs[item], self.labels[item]

def collate(data):
    indices,  protein_mols, graphs, labels = map(list, zip(*data))
    bg = dgl.batch_hetero(graphs)
    for nty in bg.ntypes:
        bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bg.canonical_etypes:
        bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)
    labels = torch.stack(labels, dim=0)

    return indices,   protein_mols, bg, labels

if __name__ == '__main__':
    import glob

    mol=Molecule("../../../geom/xyzq/benzene_2.xyzq")

    indices = filter_out_hydrogens(mol)
    atom_nums=get_atomic_numbers(mol,indices)
    print(atom_nums)


    dataset = XYZDataSet(glob.glob("../../../geom/xyzq/*.xyzq"))
    train_loader = DataLoader(dataset=dataset,
                              batch_size=2,
                              shuffle=False,
                              collate_fn=collate)

    for i_batch, sample_batched in enumerate(train_loader):
        print("batch",i_batch, len(sample_batched), sample_batched[2] ,sample_batched[3].shape)
