import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.environ.get("WRITANSUB_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
LOG_DIR = os.environ.get("WRITANSUB_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
