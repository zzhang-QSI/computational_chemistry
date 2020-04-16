import dgl
import numpy as np
import random

import rdkit
import torch
import torch.nn.functional as F
import os,sys
from dgl import model_zoo
from dgl.data.chem import PDBBind, RandomSplitter, ScaffoldSplitter, SingleTaskStratifiedSplitter
from dgl.data.utils import Subset
from itertools import accumulate
from scipy.stats import pearsonr
from torch.optim import Adam
from torch.utils.data import DataLoader
import  tqdm

from dllib.SchNet_energy import SchNet_energy,hetero2homo_graph
import  mmlib.molecule

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

from dgl.data.chem import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
def hetero2homo_graph(hetero_graph,mol):
    rd_mol = mol
    rdkit.Chem.SanitizeMol(rd_mol)
    atom_featurizer = CanonicalAtomFeaturizer()
    ### convert DGL hetero graph to DGL graph
    homo_graph = dgl.DGLGraph()
    homo_graph.from_networkx(hetero_graph.to_networkx())
    homo_graph.ndata.update(atom_featurizer(rd_mol))
    homo_graph.edata['e'] = hetero_graph.edata['distance']
    return homo_graph



def collate2(data):
    indices, protein_mols_list, ligand_mols_list, complex_mols_list, labels = map(list, zip(*data))
    labels = torch.stack(labels, dim=0)

    return indices, protein_mols_list, ligand_mols_list, complex_mols_list, labels



class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.
    Parameters
    ----------
    torch.float32 tensor of shape (T)
        Mean of existing training labels across tasks, T for the number of tasks
    torch.float32 tensor of shape (T)
        Std of existing training labels across tasks, T for the number of tasks
    """
    def __init__(self, mean=None, std=None):
        self.y_pred = []
        self.y_true = []

        if (type(mean) != type(None)) and (type(std) != type(None)):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def _finalize_labels_and_prediction(self):
        """Concatenate the labels and predictions.
        If normalization was performed on the labels, undo the normalization.
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return y_pred, y_true

    def pearson_r2(self):
        """Compute squared Pearson correlation coefficient
        Returns
        -------
        float
        """
        y_pred, y_true = self._finalize_labels_and_prediction()

        return pearsonr(y_true[:, 0].numpy(), y_pred[:, 0].numpy())[0] ** 2

    def mae(self):
        """Compute MAE
        Returns
        -------
        float
        """
        y_pred, y_true = self._finalize_labels_and_prediction()

        return F.l1_loss(y_true, y_pred).data.item()

    def rmse(self):
        """
        Compute RMSE
        Returns
        -------
        float
        """
        y_pred, y_true = self._finalize_labels_and_prediction()

        return np.sqrt(F.mse_loss(y_pred, y_true).cpu().item())

    def compute_metric(self, metric_name):
        """Compute metric
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        Returns
        -------
        float
            Metric value
        """
        assert metric_name in ['pearson_r2', 'mae', 'rmse'], \
            'Expect metric name to be "pearson_r2", "mae" or "rmse", got {}'.format(metric_name)
        if metric_name == 'pearson_r2':
            return self.pearson_r2()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'rmse':
            return self.rmse()

def update_msg_from_scores(msg, scores):
    for metric, score in scores.items():
        msg += ', {} {:.4f}'.format(metric, score)
    return msg


def mol2energy(mol):

    mmol=mmlib.molecule.Molecule(mol)
    mmol.GetEnergy()
    return mmol.e_potential
    res = rdkit.Chem.AllChem.MMFFOptimizeMoleculeConfs(mol)
    return res[0][1]
    ff = rdkit.Chem.AllChem.UFFGetMoleculeForceField(mol)
    ff.Initialize()
    ff.Minimize()

    return ff.CalcEnergy()
def pred_binding_energy_for_batch(protein_mols_list, ligand_mols_list, complex_mols_list):
    from torch.multiprocessing import Pool,cpu_count
    with Pool(8) as pool:
        protein_energy = torch.FloatTensor(pool.map(mol2energy ,protein_mols_list ))# [ mol2energy(mol) for mol in protein_mols_list])
        ligand_energy =  torch.FloatTensor(  pool.map(mol2energy ,ligand_mols_list ))#[ mol2energy(mol) for mol in ligand_mols_list])
        complex_energy = torch.FloatTensor( pool.map(mol2energy ,complex_mols_list )) #[ mol2energy(mol) for mol in complex_mols_list])


    pred_binding_energy = complex_energy - (ligand_energy + protein_energy)
    return pred_binding_energy


ACNN_PDBBind_refined_pocket_temporal = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 350,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'temporal'
}

ACNN_PDBBind_core_pocket_temporal = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 80,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'temporal'
}


class PDBBind2(PDBBind):
    def __init__(self,*args, **kwargs):
        super(PDBBind2, self).__init__(*args, **kwargs)

        self.complex_mols = dict()


    def __getitem__(self, item):
        """Get the datapoint associated with the index.
        """
        if item not in self.complex_mols:
            complex_mol = rdkit.Chem.rdmolops.CombineMols(self.protein_mols[item],self.ligand_mols[item] )
            self.complex_mols[item]=complex_mol

        return item, self.protein_mols[item], self.ligand_mols[item], \
               self.complex_mols[item], self.labels[item]


if __name__ == '__main__':
    split_dataset_cache="PDBBind_cache.pt"
    experiment_config=dict()
    experiment_config['core']=ACNN_PDBBind_core_pocket_temporal
    experiment_config['refined'] =ACNN_PDBBind_refined_pocket_temporal
    subset='refined'
    subset = 'core'

    device=torch.device("cpu")
    metrics=['pearson_r2', 'mae']
    set_random_seed(123)
    if True : #not os.path.isfile(split_dataset_cache):
        dataset = PDBBind2(subset=subset,
                          load_binding_pocket=True, sanitize=True,
                          zero_padding=True,num_processes=1)

        years = dataset.df['release_year'].values.astype(np.float32)
        indices = np.argsort(years).tolist()
        frac_list = np.array([experiment_config[subset]['frac_train'],experiment_config[subset]['frac_val'],experiment_config[subset]['frac_test'] ] )
        num_data = len(dataset)
        lengths = (num_data * frac_list).astype(int)
        lengths[-1] = num_data - np.sum(lengths[:-1])
        train_set, val_set, test_set = [
            Subset(dataset, list(indices[offset - length:offset]))
            for offset, length in zip(accumulate(lengths), lengths)]

    batch_size=experiment_config[subset]['batch_size']
    num_epochs=1
    lr=experiment_config[subset]['lr']

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate2)

    energy_model = SchNet_energy( norm=True,
                dim=experiment_config[subset]['hidden_sizes'][0],
                 output_dim=8 ).to(device)



    descriptor=tqdm.trange(num_epochs)
    train_labels = torch.stack([train_set.dataset.labels[i] for i in train_set.indices]).to(device)
    train_mean = torch.mean(train_labels)
    train_std= torch.std(train_labels)
    train_meter = Meter(train_mean, train_std)
    for epoch in descriptor:
        epoch_loss=0
        for i_batch, batch_data in enumerate(train_loader):
            indices, protein_mols_list, ligand_mols_list, complex_mols_list, labels = batch_data

            pred_binding_energy=pred_binding_energy_for_batch( protein_mols_list, ligand_mols_list, complex_mols_list)

            prediction=pred_binding_energy.view(-1,1)
            loss=torch.nn.functional.mse_loss(prediction,(labels - train_mean) / train_std)
            train_meter.update(prediction, labels)
            epoch_loss+=loss.data


        avg_loss = epoch_loss / len(train_loader.dataset)
        total_scores = {metric: train_meter.compute_metric(metric) for metric in metrics}
        msg = 'epoch {:d}/{:d}, training | loss {:.4f}'.format(
            epoch + 1,  num_epochs , avg_loss)
        msg = update_msg_from_scores(msg, total_scores)
        descriptor.set_description(msg)


    energy_model.eval()
    eval_meter = Meter(train_mean, train_std)
    test_loader = DataLoader(dataset=test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate2)
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_loader):
            indices, protein_graph_bg, ligand_graph_bg, complex_graph_bg, labels = batch_data

            pred_binding_energy=pred_binding_energy_for_batch(protein_mols_list, ligand_mols_list, complex_mols_list)

            prediction=pred_binding_energy.view(-1,1)
            eval_meter.update(prediction, labels)
    test_scores  = {metric: eval_meter.compute_metric(metric) for metric in metrics}
    test_msg = update_msg_from_scores('test results', test_scores)
    print(test_msg)