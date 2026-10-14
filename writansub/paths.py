"""存储布局（T30）：单一根目录 + 五个子目录。

根目录默认为项目目录，环境变量 WRITANSUB_HOME 整体改位置（这是唯一的路径环境变量）。
本模块 import 时不建任何目录；写入方在写之前自己 makedirs。
消费方请 `from writansub import paths` 后在调用时取 `paths.X`，不要模块级 `from writansub.paths import X`
（否则测试的 monkeypatch / reload 对它不生效）。
whisper / Qwen3 / MMS_FA 的权重走 HF、torch 各自的默认缓存，不归这里管。
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOME_DIR = os.path.abspath(os.path.expanduser(os.environ.get("WRITANSUB_HOME") or PROJECT_ROOT))

CACHE_DIR = os.path.join(HOME_DIR, "cache")    # TIGER-DnR / TIGER-Speech 的 HF 快照
MODELS_DIR = os.path.join(HOME_DIR, "models")  # 手动放置的本地模型，现只探测 Qwen3-ForcedAligner-0.6B
LOG_DIR = os.path.join(HOME_DIR, "logs")       # 会话日志 <角色>_<YYYYmmdd_HHMMSS>_<PID>.log
CONFIG_DIR = os.path.join(HOME_DIR, "config")  # 预留：第 8 步守护进程配置，现无人读写
RUN_DIR = os.path.join(HOME_DIR, "run")        # 运行期临时物：T17 溢出目录 spill_*；将来守护进程的端口 / PID 文件


def new_spill_dir() -> str:
    """在 run/ 下新建一个溢出目录并返回路径（T17 谁入口谁建谁清；T30 起不再用系统 Temp）。

    目录名带本进程 PID：强杀残留时将来的清扫器可据此判断归属，本轮不做清扫。
    """
    import tempfile
    os.makedirs(RUN_DIR, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"spill_{os.getpid()}_", dir=RUN_DIR)
