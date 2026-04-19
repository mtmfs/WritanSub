import os

from platformdirs import user_data_dir

_APP_NAME = "WritanSub"
_APP_AUTHOR = "mtmfs"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_DATA_DIR = user_data_dir(_APP_NAME, _APP_AUTHOR)
os.makedirs(USER_DATA_DIR, exist_ok=True)

PP_CONFIG_PATH = os.path.join(USER_DATA_DIR, "writansub_pp.json")
TRANSLATE_CONFIG_PATH = os.path.join(USER_DATA_DIR, "writansub_translate.json")
GUI_STATE_PATH = os.path.join(USER_DATA_DIR, "gui_state.json")

MODELS_DIR = os.environ.get("WRITANSUB_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
LOG_DIR = os.environ.get("WRITANSUB_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
