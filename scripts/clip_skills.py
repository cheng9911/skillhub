import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5
DATASET_INDEX_PATH = "dataset_index.json"
MODEL_SAVE_PATH = "openclip_finetuned.pth"

class SkillDataset(Dataset):
    def __init__(self, dataset_index_path, preprocess):
        self.samples = []  # (flan_img_path, head_img_path, skill_text, label_idx)
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

            skill_text = meta.get("description", skill_id)
            self.skill2idx[skill_id] = idx

            images_info = meta.get("images", [])
            for img_pair in images_info:
                flan_rel = img_pair.get("flan")
                head_rel = img_pair.get("head")
                if flan_rel is None or head_rel is None:
                    continue

                flan_path = os.path.join(skill_path, flan_rel)
                head_path = os.path.join(skill_path, head_rel)
                if os.path.exists(flan_path) and os.path.exists(head_path):
                    self.samples.append((flan_path, head_path, skill_text, idx))
                else:
                    print(f"Warning: missing image files for skill {skill_id}: {flan_path} or {head_path}")

        print(f"Dataset loaded: {len(self.skill2idx)} skills, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flan_path, head_path, skill_text, label = self.samples[idx]
        flan_img = Image.open(flan_path).convert("RGB")
        head_img = Image.open(head_path).convert("RGB")
        flan_img = self.preprocess(flan_img)
        head_img = self.preprocess(head_img)
        return flan_img, head_img, skill_text, label

class ClipFinetuneModel(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.text_proj = nn.Linear(clip_model.text_projection.shape[1], 512)
        self.image_proj = nn.Linear(clip_model.visual.output_dim, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, flan_imgs, head_imgs, texts, tokenizer, device):
        # 文本tokenizer是open_clip的tokenizer
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        text_features = self.text_proj(text_features)

        flan_features = self.clip_model.encode_image(flan_imgs)
        head_features = self.clip_model.encode_image(head_imgs)

        image_features = (flan_features + head_features) / 2
        image_features = self.image_proj(image_features)

        features = (image_features + text_features) / 2
        logits = self.classifier(features)
        return logits

def collate_fn(batch):
    flan_imgs, head_imgs, texts, labels = zip(*batch)
    flan_imgs = torch.stack(flan_imgs)
    head_imgs = torch.stack(head_imgs)
    labels = torch.tensor(labels)
    texts = list(texts)
    return flan_imgs, head_imgs, texts, labels

def train():
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model_clip.to(DEVICE)
    model_clip.eval()  # 冻结clip编码器，训练分类头

    dataset = SkillDataset(DATASET_INDEX_PATH, preprocess)
    num_classes = len(dataset.skill2idx)

    model = ClipFinetuneModel(model_clip, num_classes=num_classes)
    model.to(DEVICE)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for flan_imgs, head_imgs, texts, labels in dataloader:
            flan_imgs = flan_imgs.to(DEVICE)
            head_imgs = head_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(flan_imgs, head_imgs, texts, tokenizer, DEVICE)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * flan_imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/total:.4f} Accuracy: {correct/total:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
