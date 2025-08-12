import os
import json
import torch
import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "openclip_finetuned.pth"
DATASET_INDEX_PATH = "dataset_index.json"

class ClipFinetuneModel(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.text_proj = nn.Linear(clip_model.text_projection.shape[1], 512)
        self.image_proj = nn.Linear(clip_model.visual.output_dim, 512)
        self.classifier = nn.Linear(512, num_classes)

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
        logits = self.classifier(features)
        return logits

def load_skills(dataset_index_path):
    with open(dataset_index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)
    skills = dataset_index.get("skills", [])
    skill_ids = []
    skill_texts = []
    for skill in skills:
        skill_ids.append(skill["id"])
        skill_texts.append(skill["description"])
    return skill_ids, skill_texts

def preprocess_image(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # [1, C, H, W]

def predict(flan_img_path, head_img_path):
    model_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    preprocess=preprocess_train
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model_clip.to(DEVICE)
    model_clip.eval()

    skill_ids, skill_texts = load_skills(DATASET_INDEX_PATH)
    num_classes = len(skill_ids)

    model = ClipFinetuneModel(model_clip, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    flan_img = preprocess_image(flan_img_path, preprocess).to(DEVICE)
    head_img = preprocess_image(head_img_path, preprocess).to(DEVICE)

    with torch.no_grad():
        logits = model(flan_img, head_img, skill_texts, tokenizer, DEVICE)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    results = list(zip(skill_ids, skill_texts, probs))
    results.sort(key=lambda x: x[2], reverse=True)

    print("技能执行概率排序：")
    for skill_id, desc, prob in results:
        print(f"{skill_id}: {prob:.4f} -- {desc}")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python test_clip_multiskill.py <flan_image_path> <head_image_path>")
        exit(1)

    flan_img_path = sys.argv[1]
    head_img_path = sys.argv[2]
    predict(flan_img_path, head_img_path)
