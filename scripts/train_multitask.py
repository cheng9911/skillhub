#!/usr/bin/env python3
import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip
import gc

# ---------------------------
# 配置参数
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5
DATASET_INDEX_PATH = "dataset_index.json"
MODEL_SAVE_PATH = "openclip_finetuned_multitask.pth"

# ---------------------------
# 状态映射
# ---------------------------
STATUS2IDX = {
    "not_available": 0,
    "running": 1,
    "completed": 2
}
IDX2STATUS = {v: k for k, v in STATUS2IDX.items()}
NUM_STATUS = len(STATUS2IDX)

# ---------------------------
# Dataset
# ---------------------------
class SkillDataset(Dataset):
    def __init__(self, dataset_index_path, preprocess):
        self.samples = []  # (flan_img_path, head_img_path, skill_idx, status_idx)
        self.skill2idx = {}
        self.preprocess = preprocess

        with open(dataset_index_path, "r", encoding="utf-8") as f:
            dataset_index = json.load(f)

        skills = dataset_index.get("skills", [])
        for idx, skill_entry in enumerate(skills):
            skill_id = skill_entry["id"]
            skill_path = skill_entry["path"]
            meta_path = os.path.join(skill_path, "meta.json")
            if not os.path.exists(meta_path):
                print(f"Warning: skill {skill_id} missing meta.json, skipped")
                continue

            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)

            self.skill2idx[skill_id] = idx
            images_info = meta.get("images", [])

            for img_pair in images_info:
                flan_rel = img_pair.get("flan")
                head_rel = img_pair.get("head")
                status_str = img_pair.get("status", "not_available")
                if flan_rel is None or head_rel is None:
                    continue

                flan_path = os.path.join(skill_path, flan_rel)
                head_path = os.path.join(skill_path, head_rel)
                if os.path.exists(flan_path) and os.path.exists(head_path):
                    status_idx = STATUS2IDX.get(status_str, 0)
                    self.samples.append((flan_path, head_path, idx, status_idx))
                else:
                    print(f"Warning: missing image files for skill {skill_id}: {flan_path} or {head_path}")

        print(f"Dataset loaded: {len(self.skill2idx)} skills, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flan_path, head_path, skill_idx, status_idx = self.samples[idx]
        flan_img = Image.open(flan_path).convert("RGB")
        head_img = Image.open(head_path).convert("RGB")
        flan_img = self.preprocess(flan_img)
        head_img = self.preprocess(head_img)
        return flan_img, head_img, skill_idx, status_idx

# ---------------------------
# 双头模型
# ---------------------------
class ClipFinetuneMultiTaskModel(nn.Module):
    def __init__(self, clip_model, num_skills, num_status):
        super().__init__()
        self.clip_model = clip_model
        self.text_proj = nn.Linear(clip_model.text_projection.shape[1], 512)
        self.image_proj = nn.Linear(clip_model.visual.output_dim, 512)
        self.skill_classifier = nn.Linear(512, num_skills)
        self.status_classifier = nn.Linear(512, num_status)

    def forward(self, flan_imgs, head_imgs, texts, tokenizer, device):
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        text_features = self.text_proj(text_features)

        flan_features = self.clip_model.encode_image(flan_imgs)
        head_features = self.clip_model.encode_image(head_imgs)
        image_features = (flan_features + head_features) / 2
        image_features = self.image_proj(image_features)

        features = (image_features + text_features) / 2
        skill_logits = self.skill_classifier(features)
        status_logits = self.status_classifier(features)

        return skill_logits, status_logits

# ---------------------------
# Collate
# ---------------------------
def collate_fn(batch):
    flan_imgs, head_imgs, skill_labels, status_labels = zip(*batch)
    flan_imgs = torch.stack(flan_imgs)
    head_imgs = torch.stack(head_imgs)
    skill_labels = torch.tensor(skill_labels)
    status_labels = torch.tensor(status_labels)
    texts = [""] * len(flan_imgs)
    return flan_imgs, head_imgs, texts, skill_labels, status_labels

# ---------------------------
# 训练函数
# ---------------------------
def train():
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model_clip.to(DEVICE)
    model_clip.eval()

    dataset = SkillDataset(DATASET_INDEX_PATH, preprocess)
    num_skills = len(dataset.skill2idx)

    model = ClipFinetuneMultiTaskModel(model_clip, num_skills, NUM_STATUS)
    model.to(DEVICE)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_skill = 0
        correct_status = 0
        total = 0

        for flan_imgs, head_imgs, texts, skill_labels, status_labels in dataloader:
            flan_imgs = flan_imgs.to(DEVICE)
            head_imgs = head_imgs.to(DEVICE)
            skill_labels = skill_labels.to(DEVICE)
            status_labels = status_labels.to(DEVICE)

            optimizer.zero_grad()
            skill_logits, status_logits = model(flan_imgs, head_imgs, texts, tokenizer, DEVICE)

            loss_skill = criterion(skill_logits, skill_labels)
            loss_status = criterion(status_logits, status_labels)
            loss = loss_skill + loss_status

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * flan_imgs.size(0)
            preds_skill = skill_logits.argmax(dim=1)
            preds_status = status_logits.argmax(dim=1)
            correct_skill += (preds_skill == skill_labels).sum().item()
            correct_status += (preds_status == status_labels).sum().item()
            total += skill_labels.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"Loss: {total_loss/total:.4f} "
              f"Skill Acc: {correct_skill/total:.4f} "
              f"Status Acc: {correct_status/total:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    train()
