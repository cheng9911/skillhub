# skillhub
 技能库的注册和筛选

删除技能
 python scripts/delete_skill.py fold_towel
 添加技能
 python scripts/add_skill.py fold_towel "Fold Towel" "Fold a gray towel with subtle spots from its unfolded state into a neat rectangular shape." data/fold_towel/head_0000.mp4 data/fold_towel/right_0000.mp4 --max_seconds 5
 模型训练的代码
 python scripts/clip_skills.py 
 模型校验的代码
 python test_clip_multiskill.py skills/fold_towel/images/00001_flan.png skills/fold_towel/images/00001_head.png