"""预下载所有模型，避免首次运行时等待"""


def download_all():
    print("=== WritanSub 模型预下载 ===\n")

    # 1. Whisper large-v3
    print("[1/3] 下载 Whisper large-v3 ...")
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    del model
    print("      Whisper large-v3 OK\n")

    # 2. MMS_FA
    print("[2/3] 下载 MMS_FA ...")
    from torchaudio.pipelines import MMS_FA as bundle
    model = bundle.get_model()
    del model
    print("      MMS_FA OK\n")

    # 3. Cutlet (MeCab + unidic)
    print("[3/3] 初始化 Cutlet (MeCab) ...")
    import cutlet
    cutlet.Cutlet()
    print("      Cutlet OK\n")

    print("=== 全部模型准备就绪 ===")


if __name__ == "__main__":
    download_all()
