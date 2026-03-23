"""路径管理：集中定义所有运行时路径"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置文件路径
PP_CONFIG_PATH = os.path.join(PROJECT_ROOT, "aitrans_pp.json")
TRANSLATE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "aitrans_translate.json")
GUI_STATE_PATH = os.path.join(PROJECT_ROOT, "gui_state.json")

# 本地模型目录
MODELS_DIR = os.environ.get("WRITANSUB_MODELS_DIR", r"G:\本地部署模型")

# 缓存目录
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
