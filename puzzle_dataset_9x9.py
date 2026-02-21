import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Sudoku9Dataset(Dataset):
    def __init__(self, dataset_path, split="train", augment_ratio=1):
        """
        Dataset pour charger les données Sudoku 9x9 générées par build_sudoku_dataset.py.
        
        Args:
            dataset_path (str): Chemin vers le dossier racine (ex: 'data/sudoku-extreme-1k').
            split (str): 'train' ou 'test'.
        """
        super().__init__()
        self.split = split
        self.augment_ratio = augment_ratio
        
        # Structure attendue par build_sudoku_dataset.py : 
        # dataset_path/train/all__inputs.npy
        # dataset_path/train/all__labels.npy
        data_dir = os.path.join(dataset_path, split)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Le dossier {data_dir} n'existe pas.")

        inputs_path = os.path.join(data_dir, "all__inputs.npy")
        labels_path = os.path.join(data_dir, "all__labels.npy")
        
        if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
             raise FileNotFoundError(f"Fichiers .npy introuvables dans {data_dir}")

        # Chargement en mémoire (rapide pour 1k-10k exemples)
        self.inputs = np.load(inputs_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.inputs) * self.augment_ratio

    def __getitem__(self, idx):
        # Gestion de l'index pour l'augmentation virtuelle
        original_idx = idx % len(self.inputs)
        
        # Conversion en LongTensor pour PyTorch
        # Les inputs sont déjà aplatis (81,) par le script de build
        x = torch.tensor(self.inputs[original_idx], dtype=torch.long).view(-1)
        y = torch.tensor(self.labels[original_idx], dtype=torch.long).view(-1)
        
        return x, y
