import os
import json
import cv2
from datetime import datetime

SKILLS_DIR = "skills"
INDEX_FILE = "dataset_index.json"

def load_index():
    if not os.path.exists(INDEX_FILE) or os.path.getsize(INDEX_FILE) == 0:
        return {"skills": [], "last_updated": str(datetime.now().date())}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index(index):
    index["last_updated"] = str(datetime.now().date())
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

def load_meta(skill_path):
    meta_path = os.path.join(skill_path, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(skill_path, meta):
    meta_path = os.path.join(skill_path, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def get_next_frame_index(images_dir):
    existing_files = os.listdir(images_dir) if os.path.exists(images_dir) else []
    # 提取现有所有帧数字编号
    indexes = []
    for fn in existing_files:
        # 期望格式：00001_flan.png 或 00001_head.png
        if fn.endswith(".png"):
            parts = fn.split("_")
            if len(parts) == 2 and parts[0].isdigit():
                indexes.append(int(parts[0]))
    if not indexes:
        return 1
    return max(indexes) + 1

def extract_frames_dual_append(flan_video, head_video, images_dir, start_idx, max_seconds=None,frame_interval=1,seconds_per_frame=None):
    os.makedirs(images_dir, exist_ok=True)

    cap_flan = cv2.VideoCapture(flan_video)
    cap_head = cv2.VideoCapture(head_video)

    fps_flan = cap_flan.get(cv2.CAP_PROP_FPS)
    fps_head = cap_head.get(cv2.CAP_PROP_FPS)
    fps = min(fps_flan, fps_head) if fps_flan > 0 and fps_head > 0 else 30  # 默认30fps
    if seconds_per_frame is not None:
        frame_interval = int(fps * seconds_per_frame)
        print(f"⚙️ 自动换算: {seconds_per_frame} 秒/帧 -> {frame_interval} 帧间隔")
    max_frames = int(fps * max_seconds) if max_seconds is not None else None

    frame_idx = start_idx - 1
    saved_frames = []
    raw_idx = -1  # 原始帧序号
    while True:
        if max_frames is not None and (frame_idx - start_idx + 1) >= max_frames:
            break

        ret_flan, frame_flan = cap_flan.read()
        ret_head, frame_head = cap_head.read()

        if not ret_flan or not ret_head:
            break

        frame_idx += 1
        raw_idx += 1
        if raw_idx % frame_interval != 0:
            continue

        flan_path = os.path.join(images_dir, f"{frame_idx:05d}_flan.png")
        head_path = os.path.join(images_dir, f"{frame_idx:05d}_head.png")

        cv2.imwrite(flan_path, frame_flan)
        cv2.imwrite(head_path, frame_head)

        saved_frames.append({
            "flan": os.path.relpath(flan_path, os.path.dirname(images_dir)),
            "head": os.path.relpath(head_path, os.path.dirname(images_dir))
        })

    cap_flan.release()
    cap_head.release()
    print(f"✅ 追加提取完成，共保存 {len(saved_frames)} 帧")

    return saved_frames

def add_skill_append(skill_id, skill_name, skill_desc, flan_video, head_video, max_seconds=None, frame_interval=1,seconds_per_frame=None):
    skill_path = os.path.join(SKILLS_DIR, skill_id)
    images_dir = os.path.join(skill_path, "images")
    os.makedirs(skill_path, exist_ok=True)

    print(f"开始处理技能 [{skill_id}] : {skill_name}")

    old_meta = load_meta(skill_path)
    if old_meta:
        old_images = old_meta.get("images", [])
        print(f"检测到已有数据，当前已有帧数：{len(old_images)}")
    else:
        old_images = []
        print("无旧数据，初始化新技能")

    start_idx = get_next_frame_index(images_dir)

    new_frames = extract_frames_dual_append(flan_video, head_video, images_dir, start_idx, max_seconds)

    all_frames = old_images + new_frames

    meta = {
        "id": skill_id,
        "name": skill_name,
        "description": skill_desc,
        "created_at": old_meta.get("created_at") if old_meta else str(datetime.now().date()),
        "updated_at": str(datetime.now().date()),
        "images": all_frames
    }
    save_meta(skill_path, meta)
    print(f"✅ 更新 meta.json，帧数共计：{len(all_frames)}")

    index = load_index()
    index["skills"] = [s for s in index["skills"] if s["id"] != skill_id]
    index["skills"].append({
        "id": skill_id,
        "name": skill_name,
        "path": skill_path,
        "num_frames": len(all_frames),
        "description": skill_desc
    })
    save_index(index)
    print(f"✅ 更新全局索引 dataset_index.json")

if __name__ == "__main__":
    import sys

    # 简单参数解析
    max_seconds = None
    frame_interval = 1
    seconds_per_frame = None
    args = sys.argv
    if "--max_seconds" in args:
        idx = args.index("--max_seconds")
        try:
            max_seconds = float(args[idx + 1])
            # 去掉这两个参数，方便后面解析固定参数
            args = args[:idx] + args[idx+2:]
        except Exception as e:
            print("参数 --max_seconds 后应跟秒数(float)")
            exit(1)
    # 解析 --frame_interval
    if "--seconds_per_frame" in args:
        idx = args.index("--seconds_per_frame")
        try:
            seconds_per_frame = float(args[idx + 1])
            args = args[:idx] + args[idx+2:]
        except Exception:
            print("参数 --seconds_per_frame 后应跟秒数(float)")
            exit(1)
    if "--frame_interval" in args:
        idx = args.index("--frame_interval")
        try:
            frame_interval = int(args[idx + 1])
            args = args[:idx] + args[idx+2:]
        except Exception:
            print("参数 --frame_interval 后应跟整数")
            exit(1)
    if len(args) < 6:
        print("用法: python add_skill.py <skill_id> <skill_name> <skill_description> <flan_video> <head_video> [--max_seconds 秒数]")
        exit(1)

    skill_id = args[1]
    skill_name = args[2]
    skill_desc = args[3]
    flan_video = args[4]
    head_video = args[5]

    add_skill_append(skill_id, skill_name, skill_desc, flan_video, head_video, max_seconds, frame_interval,seconds_per_frame)