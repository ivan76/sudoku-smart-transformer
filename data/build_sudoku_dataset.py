from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0)).astype(np.uint8)
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board).astype(np.uint8), apply_transformation(solution).astype(np.uint8)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []
    
    print(f"\n--- Préparation du subset : {set_name} ---")
    
    csv_path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                # Validation de la longueur
                if len(q) != 81 or len(a) != 81:
                    continue
                
                # Conversion robuste : on s'assure que '.' devient 0
                # On utilise une liste en compréhension pour éviter les surprises de buffer
                q_array = np.array([int(c) if c.isdigit() else 0 for c in q.replace('.', '0')], dtype=np.uint8)
                a_array = np.array([int(c) for c in a], dtype=np.uint8)
                
                # Vérification critique : une solution ne doit JAMAIS contenir de 0
                if (a_array == 0).any():
                    # Si la solution est incomplète, on ignore cette ligne
                    continue

                inputs.append(q_array.reshape(9, 9))
                labels.append(a_array.reshape(9, 9))

    # --- Sous-échantillonnage ---
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # --- Augmentation et structuration ---
    num_augments = config.num_aug if set_name == "train" else 0
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"Augmenting {set_name}"), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        results["group_indices"].append(puzzle_id)
        
    # --- Conversion finale et Double Vérification ---
    final_inputs = np.concatenate(results["inputs"]).reshape(len(results["inputs"]), -1)
    final_labels = np.concatenate(results["labels"]).reshape(len(results["labels"]), -1)

    # ASSERTIONS DE SÉCURITÉ
    assert np.all((final_inputs >= 0) & (final_inputs <= 9)), "Erreur: Inputs hors limites (0-9)"
    assert np.all((final_labels >= 1) & (final_labels <= 9)), "Erreur: Labels invalides (doivent être 1-9)"

    results_data = {
        "inputs": final_inputs,
        "labels": final_labels,
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # --- Métadonnées alignées avec le modèle ---
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=11, # 0 à 9 + 1 pour la sécurité
        pad_id=0,
        ignore_label_id=-100, # ALIGNEMENT : On met -100 ici pour que le DataLoader le sache
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results_data["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results_data["group_indices"]) - 1,
        sets=["all"]
    )

    # Sauvegarde
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results_data.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
