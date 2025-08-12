
# skillhub

技能库的注册和筛选系统，支持技能数据采集、管理及基于双视角图像的多技能CLIP模型训练与推理。

---

## 快速开始

### 删除技能

```bash
python scripts/delete_skill.py fold_towel
```

### 添加技能

```bash
python scripts/add_skill.py fold_towel "Fold Towel" "Fold a gray towel with subtle spots from its unfolded state into a neat rectangular shape." data/fold_towel/head_0000.mp4 data/fold_towel/right_0000.mp4 --max_seconds 5
```

### 训练模型

```bash
python scripts/clip_skills.py
```

### 模型校验（推理测试）

```bash
python scripts/test_clip_multiskill.py skills/fold_towel/images/00001_flan.png skills/fold_towel/images/00001_head.png
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

---

欢迎基于本项目继续开发和优化，若有问题或建议请提交issue。

```
```
