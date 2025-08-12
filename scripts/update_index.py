import os
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
SKILL_DIR = os.path.join(ROOT, "skills")
INDEX_PATH = os.path.join(ROOT, "dataset_index.json")

def update_index():
    skills = []
    for skill_id in os.listdir(SKILL_DIR):
        skill_path = os.path.join(SKILL_DIR, skill_id)
        meta_path = os.path.join(skill_path, "meta.json")
        img_dir = os.path.join(skill_path, "images")
        
        if not os.path.isfile(meta_path):
            continue
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        num_images = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
        skills.append({
            "id": meta["id"],
            "path": f"skills/{skill_id}",
            "num_images": num_images,
            "description": meta["description"]
        })
    
    index_data = {
        "skills": skills,
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 索引已更新，当前技能数量: {len(skills)}")

if __name__ == "__main__":
    update_index()
