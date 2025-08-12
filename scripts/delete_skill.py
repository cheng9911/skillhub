
import os
import shutil
import json

SKILLS_DIR = "skills"
INDEX_FILE = "dataset_index.json"

def load_index():
    if not os.path.exists(INDEX_FILE) or os.path.getsize(INDEX_FILE) == 0:
        return {"skills": []}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index(index):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

def delete_skill(skill_id):
    skill_path = os.path.join(SKILLS_DIR, skill_id)
    if not os.path.exists(skill_path):
        print(f"技能 {skill_id} 不存在，无法删除。")
        return

    # 删除技能文件夹及内容
    shutil.rmtree(skill_path)
    print(f"技能 {skill_id} 文件夹已删除。")

    # 更新索引
    index = load_index()
    new_skills = [s for s in index.get("skills", []) if s.get("id") != skill_id]
    if len(new_skills) == len(index.get("skills", [])):
        print(f"索引中未找到技能 {skill_id}。")
    else:
        index["skills"] = new_skills
        save_index(index)
        print(f"索引中技能 {skill_id} 已删除。")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python delete_skill.py <skill_id>")
        exit(1)

    skill_id = sys.argv[1]
    delete_skill(skill_id)
