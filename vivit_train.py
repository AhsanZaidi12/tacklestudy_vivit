import os
import numpy as np
import pandas as pd
import torch
import time
import gc
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from transformers import (
    TrainingArguments,
    VivitConfig,
    VivitForVideoClassification,
    VivitImageProcessor,
    Trainer,
    TrainerCallback
)
from torch.utils.data import Dataset

IMAGE_SIZE = 224
NUM_FRAMES = 32
BATCH_SIZE = 2
NUM_CLASSES = 2
NUM_EPOCHS = 10
MODEL_CHECKPOINT = "google/vivit-b-16x2-kinetics400"
BASE_RUNS_DIR = r"/homes/ahsanzaidi/Tackle Ablation/taguchi_runs"



class LossTracker(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
            if "eval_accuracy" in logs:
                self.eval_accuracies.append(logs["eval_accuracy"])
            self.steps.append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.train_losses:
            plt.figure()
            plt.plot(self.steps[:len(self.train_losses)], self.train_losses, label="Training Loss")
            if self.eval_losses:
                plt.plot(self.steps[:len(self.eval_losses)], self.eval_losses, label="Validation Loss")
            if self.eval_accuracies:
                plt.plot(self.steps[:len(self.eval_accuracies)], self.eval_accuracies, label="Validation Accuracy")
            plt.xlabel("Steps")
            plt.ylabel("Metric Value")
            plt.title("Training & Validation Metrics")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "training_validation_metrics.png"))
            plt.close('all')

def load_video(path, num_frames=NUM_FRAMES, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
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

    if len(frames) < num_frames:
        padding = [np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                   for _ in range(num_frames - len(frames))]
        frames.extend(padding)
    elif len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    frames = np.stack(frames, axis=0)
    return frames

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = load_video(video_path)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

class VideoClassificationTrainer(Trainer):
    def __init__(self, *args, train_dataset=None, **kwargs):
        super().__init__(*args, train_dataset=train_dataset, **kwargs)
        labels = [label for label in train_dataset.labels]
        class_counts = np.bincount(labels)
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        alpha = self.class_weights[1].item()
        loss_fct = FocalLoss(alpha=alpha, gamma=2.0)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def evaluate_and_log(trainer, val_dataset, processor, name="default"):
    model = trainer.model
    model.eval()
    output_dir = os.path.join(BASE_RUNS_DIR, name, "results")
    os.makedirs(output_dir, exist_ok=True)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for sample in val_dataset:
            input_tensor = sample['pixel_values'].unsqueeze(0).to(model.device)
            label = sample['label'].item()
            outputs = model(pixel_values=input_tensor)
            pred = torch.argmax(outputs.logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)

    report = classification_report(all_labels, all_preds, target_names=["safe", "risky"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["safe", "risky"], yticklabels=["safe", "risky"])
    plt.title("Confusion Matrix (Risky = Positive)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close('all')

    risky_recall = report['risky']['recall'] if 'risky' in report else 0.0
    return report, risky_recall

def train_model(train_dataset, val_dataset, processor, name):
    output_dir = os.path.join(BASE_RUNS_DIR, name, "results")
    config = VivitConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=NUM_CLASSES)
    model = VivitForVideoClassification.from_pretrained(MODEL_CHECKPOINT, config=config, ignore_mismatched_sizes=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        dataloader_num_workers=14,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=True,
    )
    trainer = VideoClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=lambda eval_pred: {"accuracy": (np.argmax(eval_pred[0], axis=1) == eval_pred[1]).mean()},
        callbacks=[LossTracker(output_dir)]
    )
    trainer.train()
    return trainer, model

def main():
    all_summary = []
    risky_recalls = []

    for run_name in sorted(os.listdir(BASE_RUNS_DIR)):
        run_path = os.path.join(BASE_RUNS_DIR, run_name)
        if not os.path.isdir(run_path) or not run_name.startswith("run_"):
            continue

        print(f"\n🔁 Running {run_name}")
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
                fname = row['fname']
                label = int(row['label'])
                full_path = os.path.join(video_dir, fname)
                if os.path.exists(full_path):
                    paths.append(full_path)
                    labels.append(label)
            return VideoDataset(paths, labels, processor)

        train_dataset = get_dataset(train_df, train_videos)
        val_dataset = get_dataset(val_df, val_videos)

        trainer, model = train_model(train_dataset, val_dataset, processor, name=run_name)
        report, risky_recall = evaluate_and_log(trainer, val_dataset, processor, name=run_name)

        risky_recalls.append({"run": run_name, "risky_recall": risky_recall})
        summary = {
            "run": run_name,
            "safe_precision": report["safe"]["precision"],
            "safe_recall": report["safe"]["recall"],
            "safe_f1": report["safe"]["f1-score"],
            "risky_precision": report["risky"]["precision"],
            "risky_recall": report["risky"]["recall"],
            "risky_f1": report["risky"]["f1-score"],
            "accuracy": report["accuracy"]
        }
        all_summary.append(summary)

        del trainer
        del model
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        gc.collect()
        plt.close('all')

    df = pd.DataFrame(all_summary)
    risky_df = pd.DataFrame(risky_recalls)
    summary_path = os.path.join(BASE_RUNS_DIR, "taguchi_summary.xlsx")
    with pd.ExcelWriter(summary_path) as writer:
        df.to_excel(writer, index=False, sheet_name="full_summary")
        risky_df.to_excel(writer, index=False, sheet_name="risky_recall")

if __name__ == "__main__":
    main()
