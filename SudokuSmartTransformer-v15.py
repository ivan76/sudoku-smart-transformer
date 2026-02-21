from datetime import datetime
import math
from typing import Dict, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.func import functional_call, vmap

import os
from tqdm import tqdm
import copy

from CSVLogger import CSVLogger
from puzzle_dataset_9x9 import Sudoku9Dataset

# laisser Nvidia choisir les meilleurs noyaux de convolution/attention. Utile dans mon cas ?
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512   
D_MODEL = 128      
NUM_ITER = 18
LR = 2e-4
WEIGHT_DECAY = 0.02
IGNORE_LABEL_ID = -100

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

class SudokuLossHead(nn.Module):
    def __init__(self, model: nn.Module, total_warmup_steps=10000):
        super().__init__()
        self.model = model
        self.total_warmup_steps = total_warmup_steps
        self.current_step = 0 # Sera mis à jour depuis la boucle d'entraînement

    def _get_curriculum_rates(self):
        # Progression linéaire du taux d'injection (de 0.30 à 0.05)
        alpha = min(1.0, self.current_step / (self.total_warmup_steps * 6))
        # Injection : on commence à 15%, on finit à 5%
        add_rate = 0.15 + alpha * (0.05 - 0.15)
        # Dropout (masquage) : on peut rester constant à 5% ou augmenter progressivement
        drop_rate = 0.05 
        return add_rate, drop_rate

    def _augment_gpu(self, inputs, labels):
        """Augmentation structurelle complète pour Sudoku 9x9"""
        B = inputs.size(0)
        device = inputs.device
        
        # On repasse en 2D (B, 9, 9)
        inputs = inputs.view(B, 9, 9)
        labels = labels.view(B, 9, 9)

        # 1. Permutation des bandes (3 bandes horizontales de 3 lignes)
        p_bandes = torch.randperm(3, device=device)
        inputs = torch.cat([inputs[:, i*3:(i+1)*3, :] for i in p_bandes], dim=1)
        labels = torch.cat([labels[:, i*3:(i+1)*3, :] for i in p_bandes], dim=1)

        # 2. Permutation des lignes à l'intérieur de chaque bande
        row_perm_parts = []
        for i in range(3):
            row_perm_parts.append(torch.randperm(3, device=device) + (i * 3))
        row_perm = torch.cat(row_perm_parts)
        inputs = inputs[:, row_perm, :]
        labels = labels[:, row_perm, :]

        # 3. Permutation des colonnes au sein des piles verticales
        col_perm_parts = []
        for i in range(3):
            col_perm_parts.append(torch.randperm(3, device=device) + (i * 3))
        col_perm = torch.cat(col_perm_parts)
        inputs = inputs[:, :, col_perm]
        labels = labels[:, :, col_perm]

        # 4. Relabeling (Permutation des chiffres 1-9)
        perm = torch.randperm(9, device=device) + 1
        mapping = torch.cat([torch.tensor([0], device=device), perm])
        inputs = mapping[inputs.long()]
        labels = mapping[labels.long()]

        # 5. Injection/Dropout (Curriculum)
        add_rate, drop_rate = self._get_curriculum_rates()
        mask_drop = (torch.rand_like(inputs.float()) < drop_rate) & (inputs > 0)
        inputs[mask_drop] = 0
        mask_add = (torch.rand_like(inputs.float()) < add_rate) & (inputs == 0)
        inputs[mask_add] = labels[mask_add]
            
        return inputs.flatten(1), labels.flatten(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        inputs = batch["inputs"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        grid_size = self.model.grid_size

        # Augmentation GPU appliquée AVANT la projection One-Hot du modèle
        if self.training:
            inputs, labels = self._augment_gpu(inputs, labels)

        # Appel du modèle (qui gère maintenant le F.one_hot en interne)
        all_outputs = self.model({"inputs": inputs})
        
        labels_flat = labels.view(-1)
        target = labels_flat.long() - 1 # -1 car labels=1..9 et target=0..8 pour CrossEntropy
        target[labels_flat == IGNORE_LABEL_ID] = IGNORE_LABEL_ID
        
        # On recalcule le masque de perte : on n'entraîne que sur les cases vides à l'origine
        # Note : inputs a pu changer suite à l'injection de l'augmentation
        loss_mask = (inputs == 0).view(-1)
        target = torch.where(loss_mask, target, torch.tensor(IGNORE_LABEL_ID, device=target.device))

        loss = 0
        total_weight = 0
        num_steps = len(all_outputs)
        
        # Perte cumulative sur toutes les itérations de raisonnement
        for i, step_output in enumerate(all_outputs):
            # Progression de la sévérité : plus on avance, plus la fin compte
            alpha = min(1.0, self.current_step / self.total_warmup_steps)
            step_weight = (i + 1) / num_steps if alpha < 0.5 else (i / num_steps)**2
            
            logits_flat = step_output.view(-1, grid_size)
            
            # --- MODIFICATION : Pénalité sur la pire erreur de la grille ---
            # 1. Calcul de la loss par cellule sans réduction (pour garder le détail)
            pixel_losses = F.cross_entropy(logits_flat, target, ignore_index=IGNORE_LABEL_ID, reduction='none')
            
            # 2. Moyenne classique (Cell-wise loss)
            valid_elements = (target != IGNORE_LABEL_ID).sum()
            mean_loss = pixel_losses.sum() / (valid_elements + 1e-6)
            
            # 3. Pénalité "Worst Cell" : on prend la perte max par grille
            # On reshape pour isoler les grilles : (Batch, 81)
            grid_losses = pixel_losses.view(inputs.size(0), -1)
            worst_cell_loss = grid_losses.max(dim=1)[0].mean()
            
            # On combine : Loss moyenne + 10% de la pire erreur pour forcer la correction des cas difficiles
            combined_step_loss = mean_loss + 0.1 * worst_cell_loss
            
            loss += step_weight * combined_step_loss
            total_weight += step_weight
            
        loss = loss / total_weight
        
        # Calcul des métriques de précision
        with torch.no_grad():
            final_output = all_outputs[-1]
            preds = torch.argmax(final_output, dim=-1) + 1
            mask_2d = (inputs == 0)
            correct_cells = (preds == labels) & mask_2d
            
            denom = mask_2d.sum() + 1e-6
            cell_acc = correct_cells.sum().float() / denom
            
            cells_to_fill = mask_2d.sum(dim=1)
            correct_per_grid = correct_cells.sum(dim=1)
            exact_acc = (correct_per_grid == cells_to_fill).float().mean()

            metrics = {
                "train_loss": loss.detach(),
                "train_cell_acc": cell_acc,
                "train_exact_acc": exact_acc,
                # On met des tenseurs à zéro pour les valeurs par défaut
                "val_loss": torch.tensor(0.0, device=loss.device), 
                "val_cell_acc": torch.tensor(0.0, device=loss.device), 
                "val_exact_acc": torch.tensor(0.0, device=loss.device)
            }

        return loss, metrics


class SudokuFactorizedAttentionBlock(nn.Module):
    """
    Bloc d'attention factorisée optimisé pour Sudoku.
    Remplace l'attention globale coûteuse par 3 attentions locales (Ligne, Colonne, Bloc).
    Complexité réduite de O(N^2) à O(N).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, grid_size, block_r, block_c, norm_first=True):
        super().__init__()
        self.grid_size = grid_size
        self.block_r = block_r
        self.block_c = block_c
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model doit être divisible par nhead"
        
        # Projections QKV partagées pour les 3 vues (Optimisation calcul)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm_first = norm_first

    def forward(self, src, mask=None):
        # src: (B, 81, D)
        # mask: ignoré (l'attention est structurelle par construction)
        x = src
        if self.norm_first:
            x = self.norm1(x)
            
        B, L, D = x.shape
        N = self.grid_size
        H = self.nhead
        E = self.head_dim
        
        # 1. Projection QKV unique (Gain de calcul vs 3 projections séparées)
        qkv = self.qkv(x) # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Fonction helper pour préparer les vues (Batch_View, H, Seq, E)
        def prepare_view(tensor, mode):
            t_2d = tensor.view(B, N, N, D)
            if mode == 'row':
                view = t_2d.view(B * N, N, D)
            elif mode == 'col':
                view = t_2d.transpose(1, 2).reshape(B * N, N, D)
            elif mode == 'block':
                n_br = N // self.block_r
                n_bc = N // self.block_c
                view = t_2d.view(B, n_br, self.block_r, n_bc, self.block_c, D) \
                           .permute(0, 1, 3, 2, 4, 5) \
                           .reshape(B * N, N, D)
            return view.view(-1, N, H, E).transpose(1, 2)

        # Fonction mathématique pure de l'attention (Compatible VMAP)
        def batched_sdpa(q, k, v):
            d_k = q.size(-1)
            # Q * K^T divisé par racine(d_k)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            # Softmax * V
            return torch.matmul(F.softmax(scores, dim=-1), v)
        
        # 2. Attention sur les 3 vues (Lignes, Colonnes, Blocs) via FlashAttention
        # --- ROWS ---
        q_r, k_r, v_r = prepare_view(q, 'row'), prepare_view(k, 'row'), prepare_view(v, 'row')
        y_r = batched_sdpa(q_r, k_r, v_r)
        y_r = y_r.transpose(1, 2).reshape(B, L, D)
        
        # --- COLS ---
        q_c, k_c, v_c = prepare_view(q, 'col'), prepare_view(k, 'col'), prepare_view(v, 'col')
        y_c = batched_sdpa(q_c, k_c, v_c)
        y_c = y_c.transpose(1, 2).reshape(B, N, N, D).transpose(1, 2).reshape(B, L, D)
        
        # --- BLOCKS ---
        q_b, k_b, v_b = prepare_view(q, 'block'), prepare_view(k, 'block'), prepare_view(v, 'block')
        y_b = batched_sdpa(q_b, k_b, v_b)
        n_br = N // self.block_r
        n_bc = N // self.block_c
        y_b = y_b.transpose(1, 2).reshape(B, n_br, n_bc, self.block_r, self.block_c, D) \
                 .permute(0, 1, 3, 2, 4, 5).reshape(B, L, D)
                 
        # 3. Agrégation et Projection
        y = (y_r + y_c + y_b) / 3.0
        y = self.out_proj(y)
        
        # Residual + Dropout
        x = src + self.dropout(y)
        if not self.norm_first:
            x = self.norm1(x)
            
        # FFN
        residual = x
        if self.norm_first:
            x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        if not self.norm_first:
            x = self.norm2(x)
            
        return x

class SudokuSmartTransformer(nn.Module):
    def __init__(
        self,
        grid_size=9,
        block_r=None,
        block_c=None,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_iterations=24,
        weight_sharing=True,
        use_attention=True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.num_iterations = num_iterations
        self.weight_sharing = weight_sharing
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        
        # vocab_size = 10 (0 pour vide + chiffres 1 à 9)
        self.vocab_size = grid_size + 1

        if block_r is None or block_c is None:
            if grid_size == 9: block_r, block_c = 3, 3
            elif grid_size == 6: block_r, block_c = 2, 3
            else: raise ValueError("Spécifier block_r et block_c")

        self.block_r = block_r
        self.block_c = block_c
        
        # --- CHANGEMENT : ONE-HOT PROJECTION ---
        # Au lieu de nn.Embedding, on projette directement le vecteur One-Hot
        self.input_projection = nn.Linear(self.vocab_size, d_model)

        # Embeddings de position et de bloc
        self.row_embed = nn.Embedding(self.grid_size, d_model)
        self.col_embed = nn.Embedding(self.grid_size, d_model)
        self.num_blocks = (grid_size // block_r) * (grid_size // block_c)
        self.block_embed = nn.Embedding(self.num_blocks, d_model)

        if self.use_attention:
            # Fonction pour créer un bloc de transformeur (potentiellement multi-couches)
            # en empilant notre couche d'attention factorisée custom.
            def make_transformer_block():
                layers = [
                    SudokuFactorizedAttentionBlock(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        grid_size=grid_size,
                        block_r=self.block_r,
                        block_c=self.block_c,
                        norm_first=True
                    ) for _ in range(num_layers)
                ]
                return nn.Sequential(*layers)

            if self.weight_sharing:
                self.transformer_block = make_transformer_block()
            else:
                self.transformer_blocks = nn.ModuleList([
                    make_transformer_block() for _ in range(num_iterations)
                ])
        else:
            # Bloc MLP sans attention
            def make_mlp_block():
                return nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout),
                )

            if self.weight_sharing:
                self.mlp_block = make_mlp_block()
            else:
                self.mlp_blocks = nn.ModuleList([
                    make_mlp_block() for _ in range(num_iterations)
                ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, self.grid_size) # Sortie 1-9

        # --- WEIGHT TYING OPTIMISÉ ---
        # On lie les poids du classifier aux poids de la projection d'entrée (indices 1-9)
        with torch.no_grad():
            # input_projection.weight est [d_model, vocab_size] -> [128, 10]
            # On veut les indices 1 à 9 (les chiffres) : on prend les colonnes 1:
            # On transpose pour correspondre au classifier [9, 128]
            tied_weights = self.input_projection.weight[:, 1:].t() 
            self.classifier.weight.copy_(tied_weights)

        self.update_gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)

        # On fixe le poids d'ancrage pour éviter qu'il ne s'effondre à 0
        self.register_buffer("anchor_weight", torch.tensor(0.5))
        
        self.register_buffer("block_ids", self._generate_block_ids())
    
    def _generate_block_ids(self):
        block_ids = torch.zeros(self.seq_len, dtype=torch.long)
        for i in range(self.seq_len):
            r, c = i // self.grid_size, i % self.grid_size
            block_ids[i] = (r // self.block_r) * (self.grid_size // self.block_c) + (c // self.block_c)
        return block_ids

    def forward(self, batch: Dict[str, torch.Tensor]):
        x = batch["inputs"]

        # --- ÉTAPE ONE-HOT ---
        # On convertit les entiers en vecteurs de zéros et de uns
        x_one_hot = F.one_hot(x.long(), num_classes=self.vocab_size).float() # (B, 81, 10)
        
        # Projection vers d_model
        v_anchored = self.input_projection(x_one_hot) # (B, 81, d_model)

        # Embeddings de structure
        pos = torch.arange(self.seq_len, device=x.device)
        rows, cols = pos // self.grid_size, pos % self.grid_size
        
        spatial_embed = self.row_embed(rows) + self.col_embed(cols) + self.block_embed(self.block_ids)

        # État initial
        h_init = v_anchored + spatial_embed
        h = h_init

        # Masque des indices donnés pour l'ancrage
        given_mask = (x != 0).unsqueeze(-1).float()
        
        outputs = []
        for i in range(self.num_iterations):
            if self.use_attention:
                
                if self.weight_sharing:
                    h_new = self.transformer_block(h)
                else:
                    h_new = self.transformer_blocks[i](h)
                
            else:
                if self.weight_sharing:
                    h_new = h + self.mlp_block(h)  # residual
                else:
                    h_new = h + self.mlp_blocks[i](h)

            gate = torch.sigmoid(self.update_gate(h))
            h = (1 - gate) * h + gate * h_new

            # Ancrage sur les chiffres de départ
            h = h + 0.5 * given_mask * h_init
            h = self.norm(h)

            outputs.append(self.classifier(h))
            
        return outputs

    
def create_dataloader(datapath: str, split: str, augment_ratio=1):
    dataset = Sudoku9Dataset(dataset_path=datapath, split=split, augment_ratio=augment_ratio)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=8,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader

def save_checkpoint(state, filename="checkpoint.pt"):
    """Sauvegarde complète de l'état de l'entraînement."""
    print(f"-> Sauvegarde du checkpoint : {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """
    Charge un checkpoint et restaure l'état exact de tous les composants.
    """
    print(f"-> Chargement du checkpoint : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Gestion du préfixe torch.compile ('_orig_mod.')
    state_dict = checkpoint['model_state_dict']

    # Nettoyage des préfixes compile
    uncompiled_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Correction : Charger dans le modèle interne si enveloppé par SudokuLossHead
    target_model = model.model if hasattr(model, 'model') else model
        
    if hasattr(target_model, '_orig_mod'):
        target_model._orig_mod.load_state_dict(uncompiled_state_dict)
    else:
        target_model.load_state_dict(uncompiled_state_dict)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch']

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (
        min_ratio
        + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    )

def compute_lr(base_lr: float, total_steps: int, step: int):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=step,
        base_lr=base_lr,
        num_warmup_steps=round(2000),
        num_training_steps=total_steps,
        min_ratio=1.0 # 1.0 : lr constant après warmup
    )

def train_recursive_model(model, train_loader, val_loader, epochs=50000, eval_interval=1000, target_acc=99.0, resume_path=None):
    # Wrap model with loss head
    model = SudokuLossHead(model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(f"sudoku_train_metrics_{timestamp}.csv")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True, betas=(0.9, 0.95))

    scaler = GradScaler(device=DEVICE, enabled=False)

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0
    start_epoch = 0

    # --- LOGIQUE DE REPRISE  ---
    if resume_path and os.path.isfile(resume_path):
        last_epoch = load_checkpoint(resume_path, model, optimizer, scaler)
        start_epoch = last_epoch + 1
        # On recalcule le step global pour le LR scheduler
        step = start_epoch * len(train_loader)
        print(f"Reprise à partir de l'époque : {start_epoch}, step: {step}")
    else:
        step = 0

    # Paramètres pour le LR scheduler
    total_steps = len(train_loader) * epochs

    print("Setup EMA")
    ema_helper = EMAHelper()
    ema_helper.register(model)

    print(f"Entraînement lancé sur {DEVICE} en 'BF16' pour {epochs} époques.")

    # Boucle principale par intervalle d'évaluation
    for cycle_start_epoch in range(start_epoch, epochs, eval_interval):
        cycle_end_epoch = min(cycle_start_epoch + eval_interval, epochs)
        
        # Si la reprise se fait sur une époque qui est la fin d'un intervalle, on saute.
        if cycle_start_epoch >= cycle_end_epoch:
            continue

        total_intervals = math.ceil(epochs / eval_interval)
        current_interval_num = (cycle_start_epoch // eval_interval) + 1

        # Barre de progression pour l'intervalle courant
        pbar = tqdm(range(cycle_start_epoch, cycle_end_epoch), 
                    desc=f"Intervalle {current_interval_num}/{total_intervals}", 
                    leave=True)

        last_loss = 0.0
        for epoch in pbar:
            model.train()
            
            for inputs, labels in train_loader:
                batch = {"inputs": inputs, "labels": labels}
                step += 1
                model.current_step = step # On informe le LossHead du progrès

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16): # BF16 natif sur RTX PRO 6000
                    loss, metrics = model(batch=batch)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                lr_this_step = compute_lr(LR, total_steps, step)

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_this_step

                scaler.step(optimizer)
                scaler.update()

                ema_helper.update(model)
                
                # Enregistrement sans bloquer
                if step % 100 == 0:
                    # On ajoute la valeur actuelle de l'anchor_weight aux métriques
                    metrics['anchor_weight'] = model.model.anchor_weight.item()
                    add_rate, _ = model._get_curriculum_rates()
                    metrics['curriculum_add_rate'] = add_rate
                    logger.log(step, metrics)
            
            last_loss = loss.item()
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        # --- Évaluation à la fin de l'intervalle ---
        model_eval = ema_helper.ema_copy(model)
        val_results = eval_recursive(model_eval, val_loader)
        
        # On ajoute aussi l'anchor_weight (sa version EMA) aux logs de validation
        val_results['anchor_weight'] = model_eval.model.anchor_weight.item()

        # On log la validation sur le même step que l'entraînement
        logger.log(step, val_results)
        logger.flush() # Sécurité pour ne pas perdre de données si ça crash

        current_val_acc = val_results["val_exact_acc"] * 100
        
        # Affichage propre des résultats sous la barre de progression
        print(f"Fin de l'intervalle (Époque {epoch+1}) | Val Loss: {val_results['val_loss']:.4f} | Val Acc: {current_val_acc:.2f}%")

        # Stockage pour les courbes
        history["train_loss"].append(last_loss)
        history["val_acc"].append(current_val_acc)

        if current_val_acc >= target_acc:
            print(f"\nObjectif atteint ! ({current_val_acc:.2f}%). Arrêt de l'entraînement.")
            # Sauvegarde finale
            model_tosave = ema_helper.ema_copy(model)
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_tosave.model._orig_mod.state_dict() if hasattr(model_tosave.model, '_orig_mod') else model_tosave.model.state_dict(), # Pour éviter que les fichiers .pt ne contiennent les préfixes _orig_mod. (ce qui rend les checkpoints difficiles à utiliser sans torch.compile)
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, filename=f"sudoku_model_6x6_final_solved_{current_val_acc:.2f}%.pt")
            break

        if current_val_acc > best_acc:
            best_acc = current_val_acc
            model_tosave = ema_helper.ema_copy(model)
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_tosave.model._orig_mod.state_dict() if hasattr(model_tosave.model, '_orig_mod') else model_tosave.model.state_dict(), # Pour éviter que les fichiers .pt ne contiennent les préfixes _orig_mod. (ce qui rend les checkpoints difficiles à utiliser sans torch.compile)
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, filename=f'best_model_checkpoint_{epoch}.pt')

    # --- Affichage final ---
    print("terminé !")
    return history


def eval_recursive(model_head, loader):
    model_head.eval()  # On s'assure d'être en mode eval
    total_loss = 0
    total_cell_acc = 0
    total_exact_acc = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            batch = {"inputs": inputs, "labels": labels}
            # On utilise le wrapper SudokuLossHead pour récupérer la perte et les métriques
            loss, metrics = model_head(batch=batch)
            
            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_cell_acc += metrics["train_cell_acc"] * batch_size
            total_exact_acc += metrics["train_exact_acc"] * batch_size
            count += batch_size

    metrics = {
        "train_loss": 0,
        "train_cell_acc": 0,
        "train_exact_acc": 0,
        "val_loss": total_loss / count,
        "val_cell_acc": total_cell_acc / count,
        "val_exact_acc": total_exact_acc / count
    }
    
    return metrics

def train_hyperscale_es_pytorch(model, train_loader, val_loader, epochs=1000, pop_size=112, sigma=0.05, lr=0.01):
    """
    Entraînement du SudokuSmartTransformer en utilisant les principes de HyperscaleES
    adaptés pour PyTorch (via torch.func.vmap).
    """
    model_head = SudokuLossHead(model)
    
    device = next(model_head.parameters()).device
    
    # 1. Extraction des paramètres de base pour l'optimiseur central
    base_params = {k: v for k, v in model_head.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model_head.named_buffers()}
    
    # On utilise AdamW pour appliquer les pseudo-gradients (comme optax.adamw dans HyperscaleES)
    optimizer = torch.optim.AdamW(base_params.values(), lr=lr, weight_decay=0.02)
    
    # 2. Définition de la fonction d'évaluation "Stateless" pour un seul individu
    def compute_fitness(params_dict, buffers_dict, batch):
        # On appelle le modèle de façon fonctionnelle (sans modifier l'état global)
        loss, metrics = functional_call(model_head, (params_dict, buffers_dict), args=(), kwargs={'batch': batch})
        # La fitness est l'inverse de la loss (on veut maximiser la fitness)
        return -loss, metrics

    # 3. Vectorisation : On transforme la fonction pour qu'elle accepte une population de paramètres
    # in_dims=(0, None, None) signifie : vectoriser sur la dimension 0 des paramètres, 
    # mais garder le même batch et les mêmes buffers pour toute la population.
    vmap_compute_fitness = vmap(compute_fitness, in_dims=(0, None, None))

    model_head.eval() # On désactive le dropout standard car le bruit ES fait office d'exploration
    step = 0

    print(f"Début de l'entraînement ES avec une population de {pop_size} et sigma={sigma}")

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            step += 1
            model_head.current_step = step
            batch = {"inputs": inputs, "labels": labels}

            batch = {
                "inputs": inputs.to(device, non_blocking=True), 
                "labels": labels.to(device, non_blocking=True)
            }
            
            # --- ÉTAPE CRUCIALE : Augmentation partagée ---
            # Pour que l'ES fonctionne, TOUTE la population doit être évaluée sur la même grille augmentée.
            if hasattr(model_head, '_augment_gpu'):
                inputs_aug, labels_aug = model_head._augment_gpu(batch["inputs"].to(device), batch["labels"].to(device))
                batch = {"inputs": inputs_aug, "labels": labels_aug}

            # 4. Génération de la population (Échantillonnage antithétique)
            half_pop = pop_size // 2
            noises = {}
            perturbed_params = {}
            
            for name, param in base_params.items():
                # Génération de bruit aléatoire normal (similaire à EggRoll get_nonlora_update_params)
                noise = torch.randn((half_pop, *param.shape), device=device)
                # Antithétique : on utilise le bruit et son opposé pour réduire la variance du gradient
                full_noise = torch.cat([noise, -noise], dim=0)
                noises[name] = full_noise
                
                # Ajout du bruit aux paramètres de base : \theta + \sigma * \epsilon
                perturbed_params[name] = param.unsqueeze(0) + sigma * full_noise

            # 5. Évaluation parallèle de toute la population
            with torch.no_grad():
                fitnesses, metrics = vmap_compute_fitness(perturbed_params, buffers, batch)
            
            # avg_pop_exact_acc = metrics["train_exact_acc"].mean().item() * 100

            # 6. Conversion des fitness (Rank transformation ou Standardisation)
            # Équivalent de `convert_fitnesses` dans HyperscaleES
            normalized_fitness = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

            # 7. Calcul du pseudo-gradient (ES Update) et mise à jour
            optimizer.zero_grad()
            for name, param in base_params.items():
                # Règle ES : gradient = - 1/(N * sigma) * somme(fitness_i * noise_i)
                # Le "moins" est là car l'optimiseur PyTorch MINIMISE, or on veut maximiser la fitness
                grad_estimate = - (normalized_fitness.view(-1, *([1]*param.dim())) * noises[name]).mean(dim=0) / sigma
                param.grad = grad_estimate

            optimizer.step()

            # Logging
            avg_loss = -fitnesses.mean().item()
            best_loss_in_pop = -fitnesses.max().item()
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", best_loss=f"{best_loss_in_pop:.4f}")

        # Évaluation en validation à la fin de l'époque
        val_results = eval_recursive(model_head, val_loader)
        print(f"Fin Epoch {epoch+1} | Val Loss: {val_results['val_loss']:.4f} | Val Acc: {val_results['val_exact_acc']*100:.2f}%")

if __name__ == "__main__":
    # pour debug uniquement, sinon ralentit le code
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Initialisation du modèle
    model = SudokuSmartTransformer(
        grid_size=9,
        d_model=D_MODEL,
        nhead=4,
        num_layers=2,
        dim_feedforward=D_MODEL*4,
        num_iterations=NUM_ITER,
        weight_sharing=True, # Mettre à False pour tester l'alternative "Deep"
        use_attention=True, 
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nombre de paramètres entrainables : {num_params:,}")
    print("Modèle SudokuSmartTransformer : ")
    print(model)

    epochs = 50000
    eval_interval=1000

    # Compilation du modèle
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    #     print("Modèle compilé avec succès.")

    # --- DataLoaders ---
    train_loader = create_dataloader("data/sudoku-extreme-1024", "train", augment_ratio=5)
    eval_loader = create_dataloader("data/sudoku-extreme-1024", "test")

    # Pour reprendre, spécifiez le chemin. Sinon laissez None.
    CHECKPOINT_TO_RESUME = "best_model_checkpoint_3.pt" if os.path.exists("best_model_checkpoint_3.pt") else None

    # train_recursive_model(model, train_loader, eval_loader, epochs=epochs, eval_interval=eval_interval,resume_path=CHECKPOINT_TO_RESUME)

    train_hyperscale_es_pytorch(model, train_loader, eval_loader)