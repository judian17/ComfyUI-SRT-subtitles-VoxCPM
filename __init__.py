# __init__.py

import os
import folder_paths

# --- VoxCPM 路径注册 ---
# 1. 获取 ComfyUI 的主 "models" 目录
# 我们通过获取 "checkpoints" 的路径 然后取其父目录来实现
try:
    models_dir = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
except IndexError:
    # 回退方案
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# 2. 定义我们自定义的 TTS 模型路径
tts_models_path = os.path.join(models_dir, "TTS")

# 3. 确保该目录存在
os.makedirs(tts_models_path, exist_ok=True)

# 4. 关键！将此路径注册到 ComfyUI 的全局 folder_names_and_paths
# 我们给它一个 "tts" 别名，并告诉 ComfyUI 在此路径下查找（空扩展名集表示我们不关心文件类型）
folder_paths.folder_names_and_paths["tts"] = ([tts_models_path], set())
# ------------------------

# 导入您的节点
from .srt_to_speech_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 暴露映射表
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("### Loading: VoxCPM SRT Nodes (TTS path registered) ###")