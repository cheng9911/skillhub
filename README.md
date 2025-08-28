
# skillhub
SkillHub 是一个面向机器人技能学习与推理的框架，基于 **OpenCLIP** 模型进行图像-文本联合特征学习。它支持：

- **技能识别**：从多视角图像预测当前可执行技能
- **技能状态预测**：预测技能状态 `{not_available, running, completed}`
- **双头多任务学习**：同时学习技能分类和状态分类
- 可扩展到任意机器人操作技能

## 功能特点

1. **多视角输入**：支持 `flan` 和 `head` 相机图像拼接。
2. **多任务输出**：技能分类 + 状态分类。
3. **状态分数与排序**：输出每个技能可执行概率排序和状态概率。
4. **数据驱动**：技能数据由 `meta.json` 管理，支持批量标注状态。
5. **可扩展技能库**：可以轻松添加新技能和新状态。
---

## 快速开始


### 数据准备

每个技能目录下的 meta.json 格式示例：
```json
{
  "id": "pick_panda",
  "description": "Pick up the black and white panda plush toy and place it into the box.",
  "images": [
    { "flan": "00001_flan.png", "head": "00001_head.png", "status": "running" },
    { "flan": "00002_flan.png", "head": "00002_head.png", "status": "completed" }
  ]
}
```

你可以使用 scripts/add_status_to_meta.py 对已有图像快速批量添加状态。
### 添加技能

```bash
python scripts/add_skill.py fold_towel "Fold Towel" "Fold a gray towel with subtle spots from its unfolded state into a neat rectangular shape." data/fold_towel/head_0000.mp4 data/fold_towel/right_0000.mp4 --seconds_per_frame 7
```
### 删除技能

```bash
python scripts/delete_skill.py fold_towel
```
### 训练模型

```bash
python scripts/train_multitask.py

```

### 模型校验（推理示例）

```bash
python scripts/test_clip_multiskill_states.py --flan skills/fold_towel/images/00001_flan.png --head skills/fold_towel/images/00001_head.png
输出示例：
fold_towel: 0.9997 -- Fold a gray towel with subtle spots from its unfolded state into a neat rectangular shape.
Pick_panda: 0.0003 -- Grasp the black-and-white panda toy and place it into the box.

预测的技能: fold_towel (Fold a gray towel with subtle spots from its unfolded state into a neat rectangular shape.)，置信度 0.9997
预测的技能状态: completed
状态分数: {'ready': 0.0018348358571529388, 'running': 0.00046369756455533206, 'completed': 0.9977014660835266}
```
### 更新技能状态
```
 python scripts/update_meta_status.py     --meta skills/fold_towel/meta.json     --ready 1-150     --running 151-718     --completed 719-972
 ```
---

## 目录结构示意

```
skillhub/
├── dataset_index.json            # 技能集索引文件，管理所有技能元数据
├── skills/
│   ├── fold_towel/
│   │   ├── meta.json             # 单个技能描述与图像路径
│   │   └── images/
│   │       ├── 00001_flan.png
│   │       └── 00001_head.png
│   └── pick_panda/
│       ├── meta.json
│       └── images/
│           ├── ...
├── scripts/
│   ├── add_skill.py              # 添加技能脚本，支持视频采集转图片及标注
│   ├── delete_skill.py           # 删除技能脚本
│   ├── clip_skills.py            # CLIP模型多技能微调训练脚本
│   └── test_clip_multiskill.py       # 模型推理脚本，输入双视角图片输出技能概率
├── openclip_finetuned.pth # 模型权重文件
└── README.md                     # 本文件
```

---

## 说明

* 技能以文件夹为单位管理，每个技能文件夹中有 `meta.json` 和对应的双视角图像对。
* `dataset_index.json` 作为顶层索引，统一管理所有技能信息和路径。
* 训练和推理均基于 `open_clip` ，支持多技能分类，结合两个视角图像输入进行联合判断。
* 脚本均可灵活调用，方便你管理和扩展技能库。


---

## 依赖

* Python 3.8+
* PyTorch
* open\_clip ([https://github.com/mlfoundations/open\_clip](https://github.com/mlfoundations/open_clip))
* torchvision
* PIL (Pillow)

安装示例：

```bash
pip install torch torchvision pillow ftfy regex tqdm
pip install git+https://github.com/mlfoundations/open_clip.git
```
参考文献

OpenCLIP: https://github.com/mlfoundations/open_clip


---

欢迎基于本项目继续开发和优化，若有问题或建议请提交issue。
