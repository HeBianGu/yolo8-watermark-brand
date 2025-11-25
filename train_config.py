#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-05-31 11:03
# describe：
import json
from ultralytics import YOLO

# 从JSON文件加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    train_config = json.load(f)

# Load a model
model = YOLO("yolov8n.pt")

# 使用配置训练
model.train(**train_config)