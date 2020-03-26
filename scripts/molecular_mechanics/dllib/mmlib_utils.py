import warnings

from functools import partial
from multiprocessing import Pool
import numpy as np
try:
    import pdbfixer
    import simtk

    from mmlib.molecule import Molecule
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def get_mol_3D_coordinates(mol):
    """Get 3D coordinates of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    numpy.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. For failures in getting the conformations, None will be returned.
    """
    try:
        return np.array([x.coords for x in mol.atoms])
    except:
        warnings.warn('Unable to get conformation of the molecule.')
        return None

def load_molecule(molecule_file, use_conformation=True):
    """Load a molecule from a file.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format '.mol2', '.sdf',
        '.pdbqt', or '.pdb'.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    mol=Molecule(molecule_file)
    if use_conformation:
        coordinates = get_mol_3D_coordinates(mol)
    else:
        coordinates = None
    mol.GetEnergy()
    return mol, coordinates

def multiprocess_load_molecules(files,  use_conformation=True, num_processes=2):
    """Load molecules from files with multiprocessing.

    Parameters
    ----------
    files : list of str
        Each element is a path to a file storing a molecule, which can be of format '.xyz',

    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.
    num_processes : int or None
        Number of worker processes to use. If None,
        then we will use the number of CPUs in the systetm. Default to 2.

    Returns
    -------
    list of 2-tuples
        The first element of each 2-tuple is an RDKit molecule instance. The second element
        of each 2-tuple is the 3D atom coordinates of the corresponding molecule if
        use_conformation is True and the coordinates has been successfully loaded. Otherwise,
        it will be None.
    """
    if num_processes == 1:
        mols_loaded = []
        for i, f in enumerate(files):
            mols_loaded.append(load_molecule(
                f,  use_conformation=use_conformation))
    else:
        with Pool(processes=num_processes) as pool:
            mols_loaded = pool.map_async(partial(
                load_molecule,
                use_conformation=use_conformation), files)
            mols_loaded = mols_loaded.get()

    return mols_loaded
