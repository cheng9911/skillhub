#!/usr/bin/env python3
import os
import json
from PIL import Image
import torch
import torch.nn as nn
import open_clip
import argparse
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "openclip_finetuned_multitask.pth"
DATASET_INDEX_PATH = "dataset_index.json"

STATUS2IDX = {
    "ready": 0,
    "running": 1,
    "completed": 2
}
IDX2STATUS = {v: k for k, v in STATUS2IDX.items()}

# ---------------------------
# Dataset image loader
# ---------------------------
def load_image_pair(flan_path, head_path, preprocess):
    flan_img = Image.open(flan_path).convert("RGB")
    head_img = Image.open(head_path).convert("RGB")
    return preprocess(flan_img).unsqueeze(0), preprocess(head_img).unsqueeze(0)

# ---------------------------
# Multi-task model
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
# Load skills dictionary
# ---------------------------
def load_skill_dict(dataset_index_path):
    with open(dataset_index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)
    skill2idx = {}
    idx2skill = {}
    skill_descriptions = {}
    skills = dataset_index.get("skills", [])
    for idx, skill_entry in enumerate(skills):
        skill2idx[skill_entry["id"]] = idx
        idx2skill[idx] = skill_entry["id"]
        skill_descriptions[skill_entry["id"]] = skill_entry.get("description", "")
    return skill2idx, idx2skill, skill_descriptions

# ---------------------------
# Prediction with probabilities
# ---------------------------
def predict_verbose(flan_path, head_path):
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model_clip.to(DEVICE)
    model_clip.eval()

    skill2idx, idx2skill, skill_descriptions = load_skill_dict(DATASET_INDEX_PATH)
    num_skills = len(skill2idx)

    model = ClipFinetuneMultiTaskModel(model_clip, num_skills, len(STATUS2IDX))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    flan_img, head_img = load_image_pair(flan_path, head_path, preprocess)
    flan_img = flan_img.to(DEVICE)
    head_img = head_img.to(DEVICE)
    texts = [""]

    with torch.no_grad():
        skill_logits, status_logits = model(flan_img, head_img, texts, tokenizer, DEVICE)
        skill_probs = F.softmax(skill_logits, dim=1).squeeze(0)
        status_probs = F.softmax(status_logits, dim=1).squeeze(0)

        # 技能排序输出
        sorted_probs, sorted_idx = torch.sort(skill_probs, descending=True)
        print("能执行概率排序：")
        for idx in sorted_idx:
            skill_id = idx2skill[idx.item()]
            prob = sorted_probs[sorted_idx == idx].item()
            desc = skill_descriptions.get(skill_id, "")
            print(f"{skill_id}: {prob:.4f} -- {desc}")

        # 最终预测技能和状态
        pred_skill_idx = skill_probs.argmax().item()
        pred_status_idx = status_probs.argmax().item()
        pred_skill = idx2skill[pred_skill_idx]
        pred_skill_desc = skill_descriptions.get(pred_skill, "")
        pred_status = IDX2STATUS[pred_status_idx]

        status_dict = {k: status_probs[v].item() for k, v in STATUS2IDX.items()}

        print("\n预测的技能: {} ({})，置信度 {:.4f}".format(pred_skill, pred_skill_desc, skill_probs[pred_skill_idx].item()))
        print("预测的技能状态: {}".format(pred_status))
        print("状态分数: {}".format(status_dict))

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flan", required=True, help="flan camera image path")
    parser.add_argument("--head", required=True, help="head camera image path")
    args = parser.parse_args()

    predict_verbose(args.flan, args.head)
