import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PP_CONFIG_PATH = os.path.join(PROJECT_ROOT, "writansub_pp.json")
TRANSLATE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "writansub_translate.json")
GUI_STATE_PATH = os.path.join(PROJECT_ROOT, "gui_state.json")

MODELS_DIR = os.environ.get("WRITANSUB_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
