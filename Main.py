import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import json
import time


# ---------------------------
# 1) Transforms
# ---------------------------
class TwoCropTransform:
    """Create two random augmented views for TRAINING (SupCon)."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


# ---------------------------
# 2) Losses
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    Input:
      - features: [B, V, D]  (V views per sample; e.g., 2)
      - labels:   [B]
    Output:
      - scalar loss
    Notes:
      - Assumes features are already L2-normalized (recommended).
      - Correctly masks self-contrast across B*V features.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device

        if features.ndim != 3:
            raise ValueError(f"features must be [B, V, D], got {features.shape}")

        B, V, D = features.shape
        if labels.ndim != 1 or labels.shape[0] != B:
            raise ValueError(f"labels must be [B], got {labels.shape}")

        labels = labels.contiguous().view(B, 1)  # [B,1]

        # mask[i,j] = 1 if sample i and j share same class
        mask = torch.eq(labels, labels.T).float().to(device)  # [B,B]

        # Flatten views: contrast features = [B*V, D]
        contrast_features = features.reshape(B * V, D)

        # Anchors: we use all views as anchors -> [B*V, D]
        anchor_features = contrast_features
        anchor_count = V

        # Similarity logits: [B*V, B*V]
        # (features should be normalized; then matmul = cosine similarity)
        logits = torch.matmul(anchor_features, contrast_features.T) / self.temperature

        # Numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Expand the [B,B] class mask to [B*V, B*V]
        # Each anchor view of i matches all contrast views of j if class(i)==class(j)
        mask = mask.repeat_interleave(anchor_count, dim=0).repeat_interleave(V, dim=1)  # [B*V, B*V]

        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(logits, device=device)
        logits_mask.fill_diagonal_(0.0)

        # Apply masks
        mask = mask * logits_mask

        # Log-softmax over contrast dimension, excluding self
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Average over positives for each anchor
        pos_count = mask.sum(dim=1)  # [B*V]
        # If a class appears only once in batch, pos_count may be 0. Avoid NaNs.
        safe_pos_count = torch.clamp(pos_count, min=1.0)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / safe_pos_count  # [B*V]

        # Only include anchors that actually have positives (optional but cleaner)
        valid = pos_count > 0
        if valid.any():
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos[valid]
            return loss.mean()
        else:
            # No positives in the entire batch (rare with 3 classes + batch 64, but possible)
            return torch.tensor(0.0, device=device, dtype=features.dtype)


# ---------------------------
# 3) Model
# ---------------------------
class CE_SupCon_Model(nn.Module):
    def __init__(self, backbone_name='resnet50', latent_dim=512, proj_dim=128, num_classes=3):
        super().__init__()
        if backbone_name == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            for param in base.parameters():
                param.requires_grad = True
            self.encoder_cnn = nn.Sequential(*list(base.children())[:-2])
            feature_dim = 2048
        elif backbone_name == 'efficientnet_b0':
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            for param in base.parameters():
                param.requires_grad = True
            self.encoder_cnn = base.features
            feature_dim = 1280
        else:
            raise ValueError("Backbone not supported")

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(feature_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        feat = self.encoder_cnn(x)
        feat = self.avgpool(feat).flatten(1)
        h = self.encoder_mlp(feat)
        logits = self.classifier(h)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return logits, z, h


# ---------------------------
# 4) Train & Val Epoch Functions
# ---------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, supcon_criterion, ce_criterion, lambda_supcon, use_amp,
                    scheduler):
    model.train()
    total_loss, total_ce, total_sup, correct, total = 0.0, 0.0, 0.0, 0, 0

    for (x1, x2), y in loader:
        x1, x2, y = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits1, z1, _ = model(x1)
            logits2, z2, _ = model(x2)

            ce_loss = 0.5 * (ce_criterion(logits1, y) + ce_criterion(logits2, y))
            features = torch.stack([z1, z2], dim=1)
            sup_loss = supcon_criterion(features, y)
            loss = ce_loss + lambda_supcon * sup_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_ce += ce_loss.item() * bs
        total_sup += sup_loss.item() * bs
        correct += (logits1.argmax(1) == y).sum().item()
        total += bs

    return {"loss": total_loss / total, "ce": total_ce / total, "supcon": total_sup / total, "acc": correct / total}


def validate_one_epoch(model, loader, device, criterion):
    """Standard validation loop (No Augmentation, No SupCon, just CE/Acc)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Model returns: logits, z, h. We only need logits for Val.
            logits, _, _ = model(x)

            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return {"val_loss": total_loss / total, "val_acc": correct / total}


# ---------------------------
# 5) Main Training Loop
# ---------------------------
def run_training(
        model, train_loader, val_loader, optimizer, device,
        supcon_criterion, ce_criterion,
        num_epochs=10, lambda_supcon=0.1, use_amp=True, save_dir="training_logs",
        scheduler=None
):
    os.makedirs(save_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    best_val_acc = 0.0

    print(f"Starting training on {device}...")

    for epoch in range(1, num_epochs + 1):
        start_t = time.time()

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            supcon_criterion, ce_criterion, lambda_supcon, use_amp, scheduler
        )

        # --- Validate ---
        # Note: Validation uses standard CrossEntropy, not SupCon
        val_metrics = validate_one_epoch(model, val_loader, device, ce_criterion)

        duration = time.time() - start_t

        # --- Log ---
        log_entry = {
            "epoch": epoch,
            "time": duration,
            "lr": optimizer.param_groups[0]['lr'],
            **train_metrics,
            **val_metrics
        }
        history.append(log_entry)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"T_Loss: {train_metrics['loss']:.3f} T_Acc: {train_metrics['acc']:.3f} | "
              f"V_Loss: {val_metrics['val_loss']:.3f} V_Acc: {val_metrics['val_acc']:.3f} | "
              f"{duration:.1f}s")

        # --- Save Checkpoints ---
        # 1. Latest State
        '''torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history
        }, os.path.join(save_dir, "checkpoint_last.pth"))'''

        # 2. Best Model (Based on Validation Accuracy)
        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"--> New Best Val Acc: {best_val_acc:.4f} (Saved)")

        # 3. JSON Logs
        with open(os.path.join(save_dir, "training_stats.json"), 'w') as f:
            json.dump(history, f, indent=4)

    print("Training Complete.")


# ---------------------------
# 6) Entry Point
# ---------------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    SAVE_DIR = "my_experiment_logs"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # --- Transforms ---
    # Training: Strong Augmentation + Two Views
    base_train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_transform = TwoCropTransform(base_train_aug)

    # Validation: Standard transform (Resize -> CenterCrop -> Normalize)
    # Important: Do NOT use TwoCropTransform here.
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Load Data ---
    data_root = os.path.join("..", "dataset")

    if os.path.exists(data_root):
        # TRAIN SET
        ds_train = datasets.ImageFolder(os.path.join(data_root, 'train'), train_transform)
        loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
                                  prefetch_factor=2)

        # VAL SET
        ds_val = datasets.ImageFolder(os.path.join(data_root, 'val'), val_transform)
        loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        num_classes = len(ds_train.classes)
        print(f"Classes: {ds_train.classes}")
    else:
        print("Data not found, creating fake data for demo...")
        ds_train = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=3, transform=train_transform)
        ds_val = datasets.FakeData(size=50, image_size=(3, 224, 224), num_classes=3, transform=val_transform)
        loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        num_classes = 3

    # --- Model & Training ---
    model_name = "resnet50"

    for x in range(3):
        model = CE_SupCon_Model(backbone_name=model_name, num_classes=num_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        # We use 'OneCycleLR' or 'LinearWarmup + CosineAnnealing'
        # OneCycleLR is fantastic for fixed epoch runs (like your 100 epochs).
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            steps_per_epoch=len(loader_train),
            epochs=100,
            pct_start=0.1,  # Warmup for the first 10% of training (10 epochs)
            anneal_strategy='cos'
        )
        run_training(
            model=model,
            train_loader=loader_train,
            val_loader=loader_val,  # <--- Pass val loader
            optimizer=optimizer,
            device=DEVICE,
            supcon_criterion=SupConLoss(temperature=0.07),
            ce_criterion=nn.CrossEntropyLoss(),
            num_epochs=NUM_EPOCHS,
            save_dir=SAVE_DIR + model_name + "_" + "Run_" + str(x),
            scheduler=scheduler
        )
