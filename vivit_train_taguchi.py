"""
vivit_train_GRADCAM_FIXED_v8.py

NEW UPDATES (v8 - CRITICAL BUG FIX):
1. FIXED: Removed double unsqueeze - processor already returns batch dimension
2. FIXED: Added shape assertion for debugging
3. FIXED: Cast CAM computation to fp32 in bf16 mode for robustness
4. All v7 fixes maintained

CRITICAL FIX: VivitImageProcessor returns (1,T,C,H,W) but we were adding
another batch dim with unsqueeze(0) → (1,1,T,C,H,W) → crash!

Date: October 2025
"""
import os
import argparse
import numpy as np
import pandas as pd

# FIXED: Headless plotting safety for HPC clusters
import matplotlib
matplotlib.use("Agg")  # Must be before importing pyplot

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score
)
from torch.utils.data import WeightedRandomSampler
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from transformers import (
    TrainingArguments,
    VivitConfig,
    VivitForVideoClassification,
    VivitImageProcessor,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SEED FOR REPRODUCIBILITY
# ============================================================================
SEED = 42

def set_seed(seed=SEED):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = 224
NUM_FRAMES = 32
BATCH_SIZE = 2
NUM_CLASSES = 2
NUM_EPOCHS = 50
MODEL_CHECKPOINT = "google/vivit-b-16x2-kinetics400"
GRADIENT_ACCUMULATION_STEPS = 8

# Paths
BASE_RUNS_DIR = os.environ.get(
    "TACKLE_RUNS_DIR",
    os.path.abspath("./taguchi_runs")
)

RESULTS_BASE_DIR = os.environ.get(
    "TACKLE_RESULTS_DIR",
    os.path.abspath("./taguchi_runs_GRADCAM_RESULTS")
)
DEFAULT_FOLD = 0

# Threshold tuning strategy: 'macro_f1', 'neutral_recall', or 'cost_sensitive'
THRESHOLD_STRATEGY = 'macro_f1'  # RECOMMENDED: balanced approach
COST_FP = 2.0  # Cost of false positive (safe predicted as risky)
COST_FN = 1.0  # Cost of false negative (risky predicted as safe)

# Sampler behavior
# If True: same sampling order every epoch (perfect reproducibility but less diversity)
# If False: different sampling per epoch (standard practice, better training)
DETERMINISTIC_SAMPLER = False  # RECOMMENDED: False for better training

# ============================================================================
# NEUTRAL FOCAL LOSS
# ============================================================================
class NeutralFocalLoss(nn.Module):
    """Focal Loss with NEUTRAL parameters."""
    
    def __init__(self, alpha=0.6, gamma=1.6, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        print(f"\n  Using NEUTRAL Focal Loss:")
        print(f"    Alpha: {self.alpha:.2f} (neutral, no favoritism)")
        print(f"    Gamma: {self.gamma:.1f} (moderate focusing)")
        print(f"    Strategy: Balanced sampler + neutral loss\n")
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device, dtype=torch.float32),
            torch.tensor(1 - self.alpha, device=targets.device, dtype=torch.float32)
        )
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# CALLBACKS
# ============================================================================
class EnhancedLossTracker(TrainerCallback):
    """Track training metrics."""
    
    def __init__(self, output_dirs):
        self.output_dirs = output_dirs
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.eval_risky_recalls = []
        self.eval_safe_recalls = []
        self.eval_risky_precision = []
        self.eval_safe_precision = []
        self.eval_risky_f1 = []
        self.eval_safe_f1 = []
        self.eval_macro_f1 = []
        self.steps = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(state.global_step)
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.epochs.append(state.epoch if state.epoch else 0)
            if "eval_accuracy" in logs:
                self.eval_accuracies.append(logs["eval_accuracy"])
            if "eval_risky_recall" in logs:
                self.eval_risky_recalls.append(logs["eval_risky_recall"])
            if "eval_safe_recall" in logs:
                self.eval_safe_recalls.append(logs["eval_safe_recall"])
            if "eval_risky_precision" in logs:
                self.eval_risky_precision.append(logs["eval_risky_precision"])
            if "eval_safe_precision" in logs:
                self.eval_safe_precision.append(logs["eval_safe_precision"])
            if "eval_risky_f1" in logs:
                self.eval_risky_f1.append(logs["eval_risky_f1"])
            if "eval_safe_f1" in logs:
                self.eval_safe_f1.append(logs["eval_safe_f1"])
            if "eval_macro_f1" in logs:
                self.eval_macro_f1.append(logs["eval_macro_f1"])

    def on_train_end(self, args, state, control, **kwargs):
        if not self.train_losses:
            return
        
        fig = plt.figure(figsize=(18, 12))
        
        # Training loss
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.steps, self.train_losses, linewidth=2, color='blue', alpha=0.7)
        ax1.set_xlabel("Steps", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_title("Training Loss", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        if self.eval_losses:
            ax2 = plt.subplot(3, 3, 2)
            ax2.plot(self.epochs, self.eval_losses, linewidth=2, color='orange', marker='o', markersize=4)
            ax2.set_xlabel("Epoch", fontsize=11)
            ax2.set_ylabel("Loss", fontsize=11)
            ax2.set_title("Validation Loss", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Accuracy
        if self.eval_accuracies:
            ax3 = plt.subplot(3, 3, 3)
            ax3.plot(self.epochs, self.eval_accuracies, linewidth=2, color='green', marker='s', markersize=4)
            ax3.set_xlabel("Epoch", fontsize=11)
            ax3.set_ylabel("Accuracy", fontsize=11)
            ax3.set_title("Validation Accuracy", fontsize=12, fontweight='bold')
            ax3.set_ylim([0, 1.05])
            ax3.grid(True, alpha=0.3)
        
        # Recall comparison
        if self.eval_risky_recalls and self.eval_safe_recalls:
            ax4 = plt.subplot(3, 3, 4)
            ax4.plot(self.epochs, self.eval_risky_recalls, 'r-o', linewidth=2, markersize=4, label='Risky')
            ax4.plot(self.epochs, self.eval_safe_recalls, 'b-s', linewidth=2, markersize=4, label='Safe')
            ax4.set_xlabel("Epoch", fontsize=11)
            ax4.set_ylabel("Recall", fontsize=11)
            ax4.set_title("Per-Class Recall", fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1.05])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Precision comparison
        if self.eval_risky_precision and self.eval_safe_precision:
            ax5 = plt.subplot(3, 3, 5)
            ax5.plot(self.epochs, self.eval_risky_precision, 'r-^', linewidth=2, markersize=4, label='Risky')
            ax5.plot(self.epochs, self.eval_safe_precision, 'b-v', linewidth=2, markersize=4, label='Safe')
            ax5.set_xlabel("Epoch", fontsize=11)
            ax5.set_ylabel("Precision", fontsize=11)
            ax5.set_title("Per-Class Precision", fontsize=12, fontweight='bold')
            ax5.set_ylim([0, 1.05])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # F1 scores
        if self.eval_risky_f1 and self.eval_safe_f1:
            ax6 = plt.subplot(3, 3, 6)
            ax6.plot(self.epochs, self.eval_risky_f1, 'r-d', linewidth=2, markersize=4, label='Risky')
            ax6.plot(self.epochs, self.eval_safe_f1, 'b-d', linewidth=2, markersize=4, label='Safe')
            ax6.set_xlabel("Epoch", fontsize=11)
            ax6.set_ylabel("F1 Score", fontsize=11)
            ax6.set_title("Per-Class F1", fontsize=12, fontweight='bold')
            ax6.set_ylim([0, 1.05])
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Macro F1
        if self.eval_macro_f1:
            ax7 = plt.subplot(3, 3, 7)
            ax7.plot(self.epochs, self.eval_macro_f1, 'purple', linewidth=2, marker='*', markersize=6)
            ax7.set_xlabel("Epoch", fontsize=11)
            ax7.set_ylabel("Macro F1", fontsize=11)
            ax7.set_title("Macro F1 (Early Stopping Metric)", fontsize=12, fontweight='bold')
            ax7.set_ylim([0, 1.05])
            ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['images'], "training_metrics.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# UTILITIES
# ============================================================================
def create_result_directories(run_name, fold_id):
    """Create organized directory structure."""
    base_path = os.path.join(RESULTS_BASE_DIR, run_name, f"fold_{fold_id}")
    dirs = {
        'base': base_path,
        'checkpoints': os.path.join(base_path, 'checkpoints'),
        'images': os.path.join(base_path, 'images'),
        'csv': os.path.join(base_path, 'csv'),
        'logs': os.path.join(base_path, 'logs'),
        'gradcam': os.path.join(base_path, 'gradcam')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def load_video(path, num_frames=NUM_FRAMES, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """
    Load video without augmentation.
    
    FIXED: Warn on zero-frame videos for debugging.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    finally:
        cap.release()

    # FIXED: Warn if no frames were read (helps identify corrupt files)
    if len(frames) == 0:
        print(f"[WARNING] No frames read from {path} - file may be corrupt or unreadable")
    
    if len(frames) < num_frames:
        padding = [np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                   for _ in range(num_frames - len(frames))]
        frames.extend(padding)
    elif len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    return np.stack(frames, axis=0).astype(np.uint8)

# ============================================================================
# DATASET
# ============================================================================
class VideoDataset(Dataset):
    """Video dataset without augmentation."""
    
    def __init__(self, video_paths, labels, processor):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"  Dataset size: {len(labels)}")
        print(f"  Class distribution: {class_dist}")
        if len(unique) == 2:
            total = len(labels)
            print(f"  Class 0 (safe):  {counts[0]:4d} ({100*counts[0]/total:5.2f}%)")
            print(f"  Class 1 (risky): {counts[1]:4d} ({100*counts[1]/total:5.2f}%)")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = load_video(video_path)
        
        inputs = self.processor(
            list(frames), 
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False
        )
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# ============================================================================
# BALANCED SAMPLER
# ============================================================================
def create_balanced_sampler(labels, deterministic=False):
    """
    Create WeightedRandomSampler for balanced batches.
    
    Args:
        labels: array of class labels
        deterministic: If True, use seeded generator (same order every epoch)
                      If False, let PyTorch reseed per epoch (standard practice)
    
    Note: deterministic=False is RECOMMENDED for better training (more diversity),
          but deterministic=True gives perfect reproducibility.
    """
    class_counts = np.bincount(labels, minlength=2)
    class_counts[class_counts == 0] = 1
    
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    
    if deterministic:
        # Seeded generator: same sampling order every epoch (perfect reproducibility)
        g = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(labels),
            replacement=True,
            generator=g
        )
        print(f"\n  Using balanced sampling (DETERMINISTIC):")
        print(f"    Same sampling order every epoch (perfect reproducibility)")
    else:
        # No generator: PyTorch reseeds per epoch (standard practice, more diversity)
        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(labels),
            replacement=True
        )
        print(f"\n  Using balanced sampling (STANDARD):")
        print(f"    Different sampling per epoch (better training diversity)")
    
    print(f"    Class weights: {class_weights}")
    
    return sampler

# ============================================================================
# CUSTOM TRAINER
# ============================================================================
class EnhancedVideoClassificationTrainer(Trainer):
    """Trainer with neutral focal loss."""
    
    def __init__(self, *args, train_dataset=None, focal_alpha=0.5, focal_gamma=1.0, **kwargs):
        super().__init__(*args, train_dataset=train_dataset, **kwargs)
        self.loss_fn = NeutralFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# METRICS (FIXED)
# ============================================================================
def compute_metrics_standard(eval_pred):
    """
    Standard metrics using threshold=0.5.
    
    FIXED: Use .predictions and .label_ids attributes for HF compatibility.
    """
    # FIXED: Access attributes instead of unpacking
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Standard argmax (threshold=0.5)
    predictions = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    risky_precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    risky_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
    risky_f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    safe_precision = precision_score(labels, predictions, pos_label=0, zero_division=0)
    safe_recall = recall_score(labels, predictions, pos_label=0, zero_division=0)
    safe_f1 = f1_score(labels, predictions, pos_label=0, zero_division=0)
    
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "risky_precision": float(risky_precision),
        "risky_recall": float(risky_recall),
        "risky_f1": float(risky_f1),
        "safe_precision": float(safe_precision),
        "safe_recall": float(safe_recall),
        "safe_f1": float(safe_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1)
    }

# ============================================================================
# CORRECTED GRAD-CAM FOR VISION TRANSFORMERS
# ============================================================================
class ViViTGradCAM:
    """
    CORRECTED Grad-CAM implementation for Vision Transformers.
    
    FIXES:
    - Correct per-token importance using einsum
    - Robust hook registration for different HF versions
    - Proper handling of HF model outputs
    - Read patch_size from model config (no hardcoding)
    - Match input dtype to model
    """
    
    def __init__(self, model, cls_is_first=True):
        """
        Initialize Grad-CAM for ViViT model.
        
        FIXED: Read patch_size and tubelet_size from model config.
        
        Args:
            model: ViViT model
            cls_is_first: Whether CLS token is first (default: True for ViT/ViViT)
        """
        self.model = model
        self.cls_is_first = cls_is_first
        self.activations = None   # (B, N, D)
        self.gradients = None     # (B, N, D)
        self.hook_handles = []
        
        # FIXED: Read from config instead of hardcoding
        self.patch_size = getattr(model.config, "patch_size", 16)
        tubelet_size = getattr(model.config, "tubelet_size", 1)
        
        # Handle tuple format (some configs use (temporal, spatial_h, spatial_w))
        if isinstance(tubelet_size, (tuple, list)):
            self.tubelet_size = tubelet_size[0]
        else:
            self.tubelet_size = tubelet_size
        
        print(f"  Grad-CAM initialized: patch_size={self.patch_size}, tubelet_size={self.tubelet_size}")
    
    def _register_hooks(self):
        """Register hooks with robust handling of different HF versions."""
        
        def fwd(module, inp, out):
            # FIXED: Handle tuple/list/ModelOutput returns
            if isinstance(out, (tuple, list)):
                out = out[0]
            try:
                # Some HF blocks return BaseModelOutput
                out = getattr(out, "last_hidden_state", getattr(out, "hidden_states", out))
            except Exception:
                pass
            self.activations = out.detach()
        
        def bwd(module, gin, gout):
            grad = gout[0]
            try:
                grad = getattr(grad, "last_hidden_state", getattr(grad, "hidden_states", grad))
            except Exception:
                pass
            self.gradients = grad.detach()
        
        # FIXED: Robust access for different HF versions
        try:
            last_block = self.model.vivit.encoder.layer[-1]
        except AttributeError:
            try:
                last_block = self.model.vivit.encoder.layers[-1]
            except AttributeError:
                raise RuntimeError("Could not find encoder layers in model structure")
        
        self.hook_handles = [
            last_block.register_forward_hook(fwd),
            last_block.register_full_backward_hook(bwd),
        ]
    
    def _remove_hooks(self):
        """Remove registered hooks."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
    
    @torch.no_grad()
    def _infer_grid(self, num_frames, image_size):
        """Infer spatial grid dimensions."""
        H = W = image_size // self.patch_size  # e.g., 224//16 = 14
        return H, W
    
    def generate(self, pixel_values, target_class, image_size=224):
        """
        Generate Grad-CAM visualization for target class.
        
        FIXED: Correct per-token importance calculation using einsum.
        FIXED: Proper temporal tubelet handling (ViViT uses tubelet_size=2)
        FIXED: Match input dtype to model
        FIXED: Cast CAM computation to fp32 for bf16 robustness
        
        Args:
            pixel_values: (B, T, 3, H, W) - 5D tensor with batch dimension
            target_class: int, the class to generate Grad-CAM for
            image_size: int, image size (default 224)
        
        Returns:
            gradcam_per_frame: (T_orig, image_size, image_size) spatial Grad-CAM maps
        """
        device = pixel_values.device
        self._register_hooks()
        self.model.train(False)
        
        # FIXED: Match input dtype to model (handles bf16/fp16/fp32)
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(model_dtype)
        
        # Enable gradients for backward pass
        with torch.enable_grad():
            pixel_values.requires_grad_(True)
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            score = logits[0, target_class]
            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)
        
        if self.activations is None or self.gradients is None:
            self._remove_hooks()
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")
        
        # FIXED: Cast to fp32 for CAM computation (prevents bf16 quantization issues)
        A = self.activations.float()  # (B, N, D) - activations from last layer
        G = self.gradients.float()    # (B, N, D) - gradients from target class
        
        # FIXED: Correct Grad-CAM math
        # Global average pooling over batch and tokens → (D,)
        w = G.mean(dim=(0, 1))
        
        # FIXED: Per-token importance using einsum (correct!)
        # cam_tokens[n] = sum_d(A[0, n, d] * w[d])
        cam_tokens = torch.einsum('n d, d -> n', A[0], w)
        cam_tokens = torch.relu(cam_tokens)
        
        # Remove CLS token if present (ViT/ViViT use CLS as first token)
        if self.cls_is_first:
            cam_tokens = cam_tokens[1:]
        
        # Normalize
        cam_tokens = cam_tokens / (cam_tokens.max() + 1e-8)
        
        # FIXED: Use tubelet_size from init (already read from config)
        T_orig = pixel_values.shape[1]  # Original number of frames (e.g., 32)
        H = W = image_size // self.patch_size  # Spatial dimensions (e.g., 14×14)
        
        # Number of tokens (without CLS) = T_eff * H * W
        # T_eff = T_orig / tubelet_size (e.g., 32/2 = 16 for ViViT-B 16×2)
        N_no_cls = cam_tokens.numel()
        T_eff = N_no_cls // (H * W)
        
        if T_eff * H * W != N_no_cls:
            self._remove_hooks()
            raise RuntimeError(
                f"Token count mismatch: have {N_no_cls} tokens, "
                f"expected multiple of {H*W} (spatial patches). "
                f"T_eff would be {N_no_cls / (H * W):.2f}"
            )
        
        # Reshape to (T_eff, H, W) - temporal tokens, not original frames!
        cam_3d = cam_tokens.view(T_eff, H, W)
        
        # Upsample spatially to (T_eff, image_size, image_size)
        cam_up = F.interpolate(
            cam_3d.unsqueeze(0), 
            size=(image_size, image_size), 
            mode='bilinear', 
            align_corners=False
        )[0]  # (T_eff, image_size, image_size)
        
        # FIXED: Expand back to per-frame CAMs to match original frames
        if T_eff != T_orig:
            # Each tubelet represents 'tubelet_size' frames
            # Repeat each CAM to match original frame count
            repeat_factor = max(1, self.tubelet_size if self.tubelet_size > 0 else (T_orig // max(1, T_eff)))
            cam_up = cam_up.repeat_interleave(repeat_factor, dim=0)[:T_orig]
            # Now cam_up is (T_orig, image_size, image_size) aligned with input frames
        
        self._remove_hooks()
        return cam_up.detach().cpu().numpy()

def generate_gradcam_visualization(model, dataset, processor, trainer, output_dirs, run_name, fold_id):
    """
    Generate Grad-CAM visualization for highest-confidence training sample.
    
    FIXED: Efficient sample selection using batched prediction
    FIXED: Proper batch dimension handling
    """
    print(f"\nGenerating Grad-CAM visualization for highest-confidence training sample...")
    
    try:
        # FIXED: Efficient batched prediction to find highest confidence
        print("  Finding highest-confidence sample (batched)...")
        model.eval()
        
        pred_output = trainer.predict(dataset)
        all_logits = pred_output.predictions
        
        if isinstance(all_logits, torch.Tensor):
            all_logits = all_logits.cpu().numpy()
        
        # Get probabilities and find max confidence sample
        all_probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
        max_probs = all_probs.max(axis=1)
        best_idx = np.argmax(max_probs)
        max_confidence = max_probs[best_idx]
        predicted_class = all_probs[best_idx].argmax()
        
        print(f"  Best sample index: {best_idx} (confidence: {max_confidence:.2%})")
        
        # Load only the best video
        video_path = dataset.video_paths[best_idx]
        true_label = dataset.labels[best_idx]
        video_frames = load_video(video_path)
        video_name = os.path.basename(video_path).replace('.mp4', '').replace('.avi', '')
        
        print(f"  Processing: {video_name}")
        print(f"    True: {'Risky' if true_label else 'Safe'}, "
              f"Pred: {'Risky' if predicted_class else 'Safe'} ({max_confidence:.2%})")
        
        # Prepare input with proper dimensions
        device = next(model.parameters()).device
        inputs = processor(
            list(video_frames), 
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False
        )
        
        # CRITICAL FIX: VivitImageProcessor already returns (1, T, 3, H, W)
        # Do NOT add another batch dimension!
        pixel_values = inputs["pixel_values"].to(device)  # Already (1, T, 3, H, W)
        
        # FIXED: More specific assertion - check both ndim and batch size
        assert pixel_values.ndim == 5 and pixel_values.shape[0] == 1, \
            f"Expected (1,T,3,H,W); got {tuple(pixel_values.shape)}"
        
        print(f"    Input shape: {tuple(pixel_values.shape)} ✓")
        
        # Generate Grad-CAM (FIXED: no need to pass patch_size, reads from config)
        gradcam = ViViTGradCAM(model, cls_is_first=True)
        gradcam_maps = gradcam.generate(pixel_values, predicted_class, image_size=IMAGE_SIZE)
        
        # FIXED: Choose key frames based on actual CAM length (handles tubelet mapping)
        cam_len = gradcam_maps.shape[0]
        if cam_len >= 5:
            # Evenly spaced frames across the video
            key_frames = np.linspace(0, cam_len - 1, 5, dtype=int).tolist()
        else:
            # If fewer than 5 frames, use all available
            key_frames = list(range(cam_len))
        
        print(f"    CAM shape: {gradcam_maps.shape}, using frames: {key_frames}")
        
        # Visualize key frames
        fig, axes = plt.subplots(len(key_frames), 3, figsize=(12, 3*len(key_frames)))
        
        for i, frame_idx in enumerate(key_frames):
            if frame_idx < len(video_frames):
                original = video_frames[frame_idx]
                cam_spatial = gradcam_maps[frame_idx]
                
                # Create heatmap
                heatmap = np.uint8(255 * cam_spatial)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
                
                axes[i, 0].imshow(original)
                axes[i, 0].set_title(f"Frame {frame_idx}", fontsize=10)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(heatmap_color)
                axes[i, 1].set_title(f"Grad-CAM", fontsize=10)
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title(f"Overlay", fontsize=10)
                axes[i, 2].axis('off')
        
        plt.suptitle(
            f"Grad-CAM (Class: {'Risky' if predicted_class else 'Safe'}): {video_name}\n"
            f"True: {'Risky' if true_label else 'Safe'} | "
            f"Pred: {'Risky' if predicted_class else 'Safe'} ({max_confidence:.2%})\n"
            f"Highest confidence training sample",
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dirs['gradcam'], f"gradcam_{video_name}.png"),
            dpi=120, bbox_inches='tight'
        )
        plt.close()
        
        print(f"  Grad-CAM saved: gradcam_{video_name}.png")
    
    except Exception as e:
        print(f"Grad-CAM visualization failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# IMPROVED THRESHOLD TUNING
# ============================================================================
def find_optimal_threshold(all_probs, all_labels, strategy='macro_f1', cost_fp=2.0, cost_fn=1.0):
    """
    Find optimal threshold using different strategies.
    
    Args:
        all_probs: (N, 2) probability array
        all_labels: (N,) true labels
        strategy: 'macro_f1', 'neutral_recall', or 'cost_sensitive'
        cost_fp: Cost of false positive (only for cost_sensitive)
        cost_fn: Cost of false negative (only for cost_sensitive)
    
    Returns:
        optimal_threshold, metrics_at_threshold
    """
    risky_probs = all_probs[:, 1]
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, risky_probs, pos_label=1)
    
    # Ensure we have valid thresholds
    if len(thresholds) == 0:
        print("Warning: No valid thresholds found, using 0.5")
        optimal_threshold = 0.5
    else:
        if strategy == 'macro_f1':
            # RECOMMENDED: Maximize macro-F1 (balanced approach)
            print(f"\n  Strategy: Macro-F1 Maximization (balanced errors)")
            
            best_macro_f1 = 0.0
            optimal_threshold = 0.5
            
            for thr in np.linspace(0.1, 0.9, 100):
                preds = (risky_probs >= thr).astype(int)
                macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    optimal_threshold = thr
            
            print(f"    Best Macro-F1: {best_macro_f1:.4f} at threshold={optimal_threshold:.3f}")
        
        elif strategy == 'neutral_recall':
            # Target 50% recall for both classes (neutral)
            print(f"\n  Strategy: Neutral Recall (target=0.5 for risky class)")
            
            target_recall = 0.5
            valid_indices = np.where(recall >= target_recall)[0]
            
            if len(valid_indices) > 0:
                # Find index with best precision among valid recalls
                best_idx = valid_indices[np.argmax(precision[valid_indices])]
                
                # BOUNDS CHECK: sklearn returns precision/recall with one extra element
                # precision[i], recall[i] correspond to thresholds[i] for i < len(thresholds)
                # precision[-1]=1, recall[-1]=0 have no corresponding threshold
                if best_idx < len(thresholds):
                    optimal_threshold = thresholds[best_idx]
                else:
                    # Last element (all positive predictions), use very low threshold
                    optimal_threshold = 0.0
                    print(f"    Warning: Best point is at 100% recall, using threshold=0.0")
            else:
                # If can't achieve target recall, maximize F1
                print(f"    Warning: Cannot achieve target recall {target_recall}, maximizing F1 instead")
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
                
                # FIXED: Add bounds check in fallback branch too
                if best_idx < len(thresholds):
                    optimal_threshold = thresholds[best_idx]
                elif best_idx == len(thresholds):
                    # Last element, use very low threshold
                    optimal_threshold = 0.0
                else:
                    # Fallback to 0.5
                    optimal_threshold = 0.5
            
            print(f"    Threshold: {optimal_threshold:.3f}")
        
        elif strategy == 'cost_sensitive':
            # Minimize cost = c_fp * FP + c_fn * FN
            print(f"\n  Strategy: Cost-Sensitive (FP_cost={cost_fp}, FN_cost={cost_fn})")
            
            min_cost = float('inf')
            optimal_threshold = 0.5
            
            for thr in np.linspace(0.1, 0.9, 100):
                preds = (risky_probs >= thr).astype(int)
                cm = confusion_matrix(all_labels, preds, labels=[0, 1])
                
                # cm[0, 1] = FP (safe predicted as risky)
                # cm[1, 0] = FN (risky predicted as safe)
                if cm.shape == (2, 2):
                    fp = cm[0, 1]
                    fn = cm[1, 0]
                    total_cost = cost_fp * fp + cost_fn * fn
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        optimal_threshold = thr
            
            print(f"    Minimum cost: {min_cost:.1f} at threshold={optimal_threshold:.3f}")
        
        else:
            print(f"Warning: Unknown strategy '{strategy}', using macro_f1")
            return find_optimal_threshold(all_probs, all_labels, strategy='macro_f1')
    
    # Calculate metrics at optimal threshold
    preds_at_threshold = (risky_probs >= optimal_threshold).astype(int)
    
    metrics_at_threshold = {
        'optimal_threshold': float(optimal_threshold),
        'strategy': strategy,
        'risky_precision': float(precision_score(all_labels, preds_at_threshold, pos_label=1, zero_division=0)),
        'risky_recall': float(recall_score(all_labels, preds_at_threshold, pos_label=1, zero_division=0)),
        'risky_f1': float(f1_score(all_labels, preds_at_threshold, pos_label=1, zero_division=0)),
        'safe_precision': float(precision_score(all_labels, preds_at_threshold, pos_label=0, zero_division=0)),
        'safe_recall': float(recall_score(all_labels, preds_at_threshold, pos_label=0, zero_division=0)),
        'safe_f1': float(f1_score(all_labels, preds_at_threshold, pos_label=0, zero_division=0)),
        'accuracy': float(accuracy_score(all_labels, preds_at_threshold)),
        'macro_f1': float(f1_score(all_labels, preds_at_threshold, average='macro', zero_division=0))
    }
    
    if strategy == 'cost_sensitive':
        cm = confusion_matrix(all_labels, preds_at_threshold, labels=[0, 1])
        if cm.shape == (2, 2):
            metrics_at_threshold['fp_count'] = int(cm[0, 1])
            metrics_at_threshold['fn_count'] = int(cm[1, 0])
            metrics_at_threshold['total_cost'] = float(cost_fp * cm[0, 1] + cost_fn * cm[1, 0])
    
    return optimal_threshold, metrics_at_threshold

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model_with_threshold_tuning(trainer, model, val_dataset, processor, output_dirs, 
                                        run_name, fold_id, threshold_strategy='macro_f1',
                                        cost_fp=2.0, cost_fn=1.0):
    """Evaluate model with improved threshold tuning."""
    model.eval()
    
    print(f"\nEvaluating on {len(val_dataset)} samples (batched)...")
    
    # Batched prediction
    pred_output = trainer.predict(val_dataset)
    all_logits = pred_output.predictions
    all_labels = pred_output.label_ids
    
    if isinstance(all_logits, torch.Tensor):
        all_logits = all_logits.cpu().numpy()
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.cpu().numpy()
    
    # Get probabilities and predictions
    all_probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    all_preds_standard = np.argmax(all_logits, axis=1)
    
    # Standard metrics
    report_standard = classification_report(
        all_labels, all_preds_standard,
        target_names=["Safe", "Risky"],
        output_dict=True,
        zero_division=0
    )
    
    print("\n  Standard (threshold=0.5) results:")
    print(f"    Safe:  P={report_standard['Safe']['precision']:.3f} R={report_standard['Safe']['recall']:.3f}")
    print(f"    Risky: P={report_standard['Risky']['precision']:.3f} R={report_standard['Risky']['recall']:.3f}")
    
    # Find optimal threshold with chosen strategy
    print(f"\nFinding optimal threshold...")
    optimal_threshold, metrics_optimal = find_optimal_threshold(
        all_probs, all_labels, strategy=threshold_strategy, cost_fp=cost_fp, cost_fn=cost_fn
    )
    
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"    Safe:  P={metrics_optimal['safe_precision']:.3f} R={metrics_optimal['safe_recall']:.3f} F1={metrics_optimal['safe_f1']:.3f}")
    print(f"    Risky: P={metrics_optimal['risky_precision']:.3f} R={metrics_optimal['risky_recall']:.3f} F1={metrics_optimal['risky_f1']:.3f}")
    print(f"    Macro-F1: {metrics_optimal['macro_f1']:.3f}")
    
    if threshold_strategy == 'cost_sensitive':
        print(f"    FP: {metrics_optimal['fp_count']}, FN: {metrics_optimal['fn_count']}, Total Cost: {metrics_optimal['total_cost']:.1f}")
    
    # Predictions with optimal threshold
    all_preds_optimal = (all_probs[:, 1] >= optimal_threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds_optimal, labels=[1, 0])
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                     xticklabels=["Risky", "Safe"],
                     yticklabels=["Risky", "Safe"],
                     cbar_kws={'label': 'Count'},
                     linewidths=3, linecolor='black')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            plt.text(j + 0.5, i + 0.5, f'{count}',
                    ha='center', va='center', 
                    fontsize=32, fontweight='bold', color='black')
    
    plt.title(f"Confusion Matrix {run_name} fold_{fold_id}\nStrategy: {threshold_strategy}", 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['images'], 
                            f"confusion_matrix_{run_name}_fold{fold_id}.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Performance curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    precision_safe, recall_safe, _ = precision_recall_curve((all_labels == 0).astype(int), all_probs[:, 0])
    ap_safe = average_precision_score((all_labels == 0).astype(int), all_probs[:, 0])
    axes[0, 0].plot(recall_safe, precision_safe, linewidth=2, color='blue')
    axes[0, 0].fill_between(recall_safe, precision_safe, alpha=0.2, color='blue')
    axes[0, 0].set_title(f'Safe PR Curve (AP={ap_safe:.3f})', fontsize=13)
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True, alpha=0.3)
    
    precision_risky, recall_risky, _ = precision_recall_curve((all_labels == 1).astype(int), all_probs[:, 1])
    ap_risky = average_precision_score((all_labels == 1).astype(int), all_probs[:, 1])
    axes[0, 1].plot(recall_risky, precision_risky, linewidth=2, color='red')
    axes[0, 1].fill_between(recall_risky, precision_risky, alpha=0.2, color='red')
    axes[0, 1].scatter([metrics_optimal['risky_recall']], [metrics_optimal['risky_precision']], 
                      s=200, c='green', marker='*', edgecolors='black', linewidths=2,
                      label=f'Optimal (thr={optimal_threshold:.3f})', zorder=5)
    axes[0, 1].set_title(f'Risky PR Curve (AP={ap_risky:.3f})', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    fpr_safe, tpr_safe, _ = roc_curve((all_labels == 0).astype(int), all_probs[:, 0])
    roc_auc_safe = auc(fpr_safe, tpr_safe)
    axes[1, 0].plot(fpr_safe, tpr_safe, linewidth=2, color='blue', label=f'AUC={roc_auc_safe:.3f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1, 0].set_title('Safe ROC', fontsize=13)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    fpr_risky, tpr_risky, _ = roc_curve((all_labels == 1).astype(int), all_probs[:, 1])
    roc_auc_risky = auc(fpr_risky, tpr_risky)
    axes[1, 1].plot(fpr_risky, tpr_risky, linewidth=2, color='red', label=f'AUC={roc_auc_risky:.3f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1, 1].set_title('Risky ROC', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Performance - {run_name}/fold_{fold_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['images'], f"performance_{run_name}_fold{fold_id}.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    metrics_summary = {
        "run": run_name,
        "fold": fold_id,
        "threshold_strategy": threshold_strategy,
        "std_safe_precision": report_standard["Safe"]["precision"],
        "std_safe_recall": report_standard["Safe"]["recall"],
        "std_safe_f1": report_standard["Safe"]["f1-score"],
        "std_risky_precision": report_standard["Risky"]["precision"],
        "std_risky_recall": report_standard["Risky"]["recall"],
        "std_risky_f1": report_standard["Risky"]["f1-score"],
        "std_accuracy": report_standard["accuracy"],
        "optimal_threshold": optimal_threshold,
        "opt_safe_precision": metrics_optimal["safe_precision"],
        "opt_safe_recall": metrics_optimal["safe_recall"],
        "opt_safe_f1": metrics_optimal["safe_f1"],
        "opt_risky_precision": metrics_optimal["risky_precision"],
        "opt_risky_recall": metrics_optimal["risky_recall"],
        "opt_risky_f1": metrics_optimal["risky_f1"],
        "opt_accuracy": metrics_optimal["accuracy"],
        "opt_macro_f1": metrics_optimal["macro_f1"],
        "safe_ap": ap_safe,
        "safe_roc_auc": roc_auc_safe,
        "risky_ap": ap_risky,
        "risky_roc_auc": roc_auc_risky,
    }
    
    if threshold_strategy == 'cost_sensitive':
        metrics_summary['fp_count'] = metrics_optimal['fp_count']
        metrics_summary['fn_count'] = metrics_optimal['fn_count']
        metrics_summary['total_cost'] = metrics_optimal['total_cost']
    
    pd.DataFrame([metrics_summary]).to_csv(
        os.path.join(output_dirs['csv'], "metrics_summary.csv"), index=False
    )
    
    return report_standard, metrics_summary

# ============================================================================
# TRAINING
# ============================================================================
def train_model_stable(train_dataset, val_dataset, processor, run_name, fold_id, 
                      focal_alpha=0.6, focal_gamma=1.6, deterministic_sampler=False):
    """Train model with all fixes applied."""
    output_dirs = create_result_directories(run_name, fold_id)
    
    config = VivitConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    )
    
    model = VivitForVideoClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Create balanced sampler
    sampler = create_balanced_sampler(train_dataset.labels, deterministic=deterministic_sampler)
    
    # Check precision support
    has_cuda = torch.cuda.is_available()
    bf16_support = has_cuda and torch.cuda.get_device_capability()[0] >= 8
    use_bf16 = bf16_support
    use_fp16 = has_cuda and not bf16_support
    
    print(f"\n  Precision: {'bf16' if use_bf16 else ('fp16' if use_fp16 else 'fp32')}")
    print(f"  Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    
    trainer_obj = EnhancedVideoClassificationTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dirs['checkpoints'],
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            eval_accumulation_steps=8,
            num_train_epochs=NUM_EPOCHS,
            dataloader_num_workers=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_dir=output_dirs['logs'],
            logging_steps=10,
            logging_first_step=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            bf16=use_bf16,
            fp16=use_fp16,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            seed=SEED,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_standard,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    # Override dataloader with balanced sampler
    def get_train_dataloader_with_sampler():
        from torch.utils.data import DataLoader
        return DataLoader(
            train_dataset,
            batch_size=trainer_obj.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=trainer_obj.args.dataloader_num_workers,
            pin_memory=trainer_obj.args.dataloader_pin_memory,
            persistent_workers=trainer_obj.args.dataloader_persistent_workers,
        )
    
    trainer_obj.get_train_dataloader = get_train_dataloader_with_sampler
    
    # Add callbacks
    loss_tracker = EnhancedLossTracker(output_dirs)
    trainer_obj.add_callback(loss_tracker)
    trainer_obj.add_callback(EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.001
    ))
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {run_name}/fold_{fold_id}")
    print(f"{'='*70}\n")
    
    trainer_obj.train()
    
    return trainer_obj, model, output_dirs

# ============================================================================
# MAIN
# ============================================================================
def main(specified_runs=None, fold_id=DEFAULT_FOLD, focal_alpha=0.6, focal_gamma=1.6, 
         threshold_strategy=THRESHOLD_STRATEGY, cost_fp=COST_FP, cost_fn=COST_FN,
         deterministic_sampler=DETERMINISTIC_SAMPLER):
    """Main training loop."""
    set_seed(SEED)
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    
    all_runs = sorted([
        d for d in os.listdir(BASE_RUNS_DIR)
        if os.path.isdir(os.path.join(BASE_RUNS_DIR, d))
        and d.startswith("run_")
        and "catalog" not in d.lower()
        and "result" not in d.lower()
    ])
    
    if specified_runs:
        runs_to_process = [r for r in all_runs if r in specified_runs]
    else:
        runs_to_process = all_runs
    
    print(f"\n{'='*70}")
    print(f"CORRECTED GRAD-CAM VERSION v8")
    print(f"{'='*70}")
    print(f"Processing {len(runs_to_process)} runs with fold_{fold_id}")
    print(f"Seed: {SEED}")
    print(f"Focal Loss: alpha={focal_alpha} (neutral), gamma={focal_gamma} (moderate)")
    print(f"Threshold Strategy: {threshold_strategy}")
    if threshold_strategy == 'cost_sensitive':
        print(f"  FP Cost: {cost_fp}, FN Cost: {cost_fn}")
    print(f"Sampler: {'Deterministic (same order every epoch)' if deterministic_sampler else 'Standard (different per epoch)'}")
    print("\nCRITICAL FIX (v8): Removed double batch dimension!")
    print("  - VivitImageProcessor returns (1,T,C,H,W)")
    print("  - Was adding unsqueeze(0) → (1,1,T,C,H,W) → CRASH")
    print("  - Now correctly using processor output as-is")
    print("\nAll Fixes Summary:")
    print("  1. FIXED: Removed double unsqueeze (v8 - CRITICAL)")
    print("  2. FIXED: Cast CAM to fp32 for bf16 robustness (v8)")
    print("  3. FIXED: Added shape assertion for debugging (v8)")
    print("  4. FIXED: Read patch_size/tubelet_size from config (v7)")
    print("  5. FIXED: Match input dtype to model (v7)")
    print("  6. FIXED: Headless plotting safety (v7)")
    print("  7. FIXED: Temporal tubelet handling (v6)")
    print("  8. FIXED: Correct Grad-CAM math using einsum (v5)")
    print("  9. FIXED: Efficient batched sample selection (v5)")
    print(" 10. Multiple threshold strategies (macro_f1 recommended)")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for run_idx, run_name in enumerate(runs_to_process, 1):
        run_path = os.path.join(BASE_RUNS_DIR, run_name, f"fold_{fold_id}")
        
        if not os.path.exists(run_path):
            print(f"\nWarning: {run_name}/fold_{fold_id} not found, skipping")
            continue
        
        print(f"\n{'='*70}")
        print(f"[{run_idx}/{len(runs_to_process)}] {run_name}/fold_{fold_id}")
        print(f"{'='*70}")
        
        train_videos = os.path.join(run_path, "train/videos")
        train_labels = os.path.join(run_path, "train_labels.csv")
        val_videos = os.path.join(run_path, "val/videos")
        val_labels = os.path.join(run_path, "val_labels.csv")
        
        processor = VivitImageProcessor.from_pretrained(MODEL_CHECKPOINT)
        train_df = pd.read_csv(train_labels)
        val_df = pd.read_csv(val_labels)
        
        def get_dataset(df, video_dir):
            paths, labels = [], []
            for _, row in df.iterrows():
                full_path = os.path.join(video_dir, row['fname'])
                if os.path.exists(full_path):
                    paths.append(full_path)
                    labels.append(int(row['label']))
            return VideoDataset(paths, labels, processor)
        
        print("\nTrain:")
        train_dataset = get_dataset(train_df, train_videos)
        print("\nVal:")
        val_dataset = get_dataset(val_df, val_videos)
        
        trainer, model, output_dirs = train_model_stable(
            train_dataset, val_dataset, processor, run_name, fold_id,
            focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            deterministic_sampler=deterministic_sampler
        )
        
        print(f"\n{'='*70}")
        print(f"EVALUATION - {run_name}/fold_{fold_id}")
        print(f"{'='*70}")
        
        report, metrics_summary = evaluate_model_with_threshold_tuning(
            trainer, model, val_dataset, processor, output_dirs, run_name, fold_id,
            threshold_strategy=threshold_strategy, cost_fp=cost_fp, cost_fn=cost_fn
        )
        
        # Generate Grad-CAM for highest-confidence training sample
        generate_gradcam_visualization(
            model, train_dataset, processor, trainer, output_dirs, run_name, fold_id
        )
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - {run_name}/fold_{fold_id}")
        print(f"{'='*70}")
        print(f"\nStandard (threshold=0.5):")
        print(f"  Safe:  Recall={metrics_summary['std_safe_recall']:.4f}")
        print(f"  Risky: Recall={metrics_summary['std_risky_recall']:.4f}")
        print(f"  Accuracy: {metrics_summary['std_accuracy']:.4f}")
        print(f"\nOptimal (threshold={metrics_summary['optimal_threshold']:.3f}, strategy={threshold_strategy}):")
        print(f"  Safe:  Recall={metrics_summary['opt_safe_recall']:.4f}")
        print(f"  Risky: Recall={metrics_summary['opt_risky_recall']:.4f}")
        print(f"  Macro-F1: {metrics_summary['opt_macro_f1']:.4f}")
        print(f"  Accuracy: {metrics_summary['opt_accuracy']:.4f}")
        print(f"{'='*70}\n")
        
        all_results.append(metrics_summary)
        
        del trainer, model, train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        plt.close('all')
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(RESULTS_BASE_DIR, f"results_fold{fold_id}_{threshold_strategy}.csv")
        results_df.to_csv(results_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Strategy: {threshold_strategy}")
        print(f"Avg Optimal Safe Recall:  {results_df['opt_safe_recall'].mean():.4f}")
        print(f"Avg Optimal Risky Recall: {results_df['opt_risky_recall'].mean():.4f}")
        print(f"Avg Optimal Macro-F1:     {results_df['opt_macro_f1'].mean():.4f}")
        print(f"\nResults saved: {results_path}")
        print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViViT training with CORRECTED Grad-CAM v8 (FIXED crash)')
    parser.add_argument('--runs', type=str, default=None, help='Comma-separated runs')
    parser.add_argument('--fold', type=int, default=DEFAULT_FOLD, help='Fold ID')
    parser.add_argument('--alpha', type=float, default=0.6, help='Focal loss alpha (0.5=neutral)')
    parser.add_argument('--gamma', type=float, default=1.6, help='Focal loss gamma (1.0=moderate)')
    parser.add_argument('--threshold_strategy', type=str, default='macro_f1', 
                       choices=['macro_f1', 'neutral_recall', 'cost_sensitive'],
                       help='Threshold tuning strategy (default: macro_f1)')
    parser.add_argument('--cost_fp', type=float, default=2.0, 
                       help='Cost of false positive (for cost_sensitive strategy)')
    parser.add_argument('--cost_fn', type=float, default=1.0, 
                       help='Cost of false negative (for cost_sensitive strategy)')
    parser.add_argument('--deterministic_sampler', action='store_true',
                       help='Use deterministic sampler (same order every epoch). Default: False (better training)')
    args = parser.parse_args()
    
    if args.runs:
        specified_runs = [r.strip() for r in args.runs.split(',')]
        main(specified_runs=specified_runs, fold_id=args.fold, 
             focal_alpha=args.alpha, focal_gamma=args.gamma,
             threshold_strategy=args.threshold_strategy,
             cost_fp=args.cost_fp, cost_fn=args.cost_fn,
             deterministic_sampler=args.deterministic_sampler)
    else:
        main(fold_id=args.fold, focal_alpha=args.alpha, focal_gamma=args.gamma,
             threshold_strategy=args.threshold_strategy,
             cost_fp=args.cost_fp, cost_fn=args.cost_fn,
             deterministic_sampler=args.deterministic_sampler)
