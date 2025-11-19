# srt_to_speech_nodes.py

import torch
import numpy as np
import os
import re
import folder_paths
import tempfile
import soundfile as sf
import logging
logger = logging.getLogger(__name__)
import time

# 导入 ComfyUI 的模型管理器
import comfy.model_management as model_management 

# ... (所有其他 import 语句保持不变) ...
try:
    # 导入本地的 "voxcpm" 包
    from . import voxcpm

    # 导入我们需要的子模块
    from .voxcpm.utils import text_normalize

    logger.info("Successfully imported local 'voxcpm' library.")

except ImportError as e:
    logger.error(f"Failed to import local 'voxcpm' library. Make sure it's copied into the node directory. Error: {e}")
    raise e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[VoxCPM-SRT-Processor] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
try:
    import pydub
    PYDUB_AVAILABLE = True
    logger.info("pydub found. High-quality time-stretching is enabled.")
except ImportError:
    PYDUB_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# --- 辅助函数：模型设备管理 (强化版) ---

def move_voxcpm_to_device(model, device):
    """将 VoxCPM 的所有已知 torch.nn.Module 和缓存移动到指定设备"""
    logger.info(f"--- Moving VoxCPM components to {device} ---")
    try:
        # 1. 移动 TTS Model (主模块)
        if hasattr(model, 'tts_model') and isinstance(model.tts_model, torch.nn.Module):
            model.tts_model.to(device)
            logger.info(f"Moved tts_model (nn.Module) to {device}")
        
        # 2. 移动 TTS Model 的 KV 缓存 (它们是 Tensors, 不是 Modules)
        try:
            if hasattr(model.tts_model, 'base_lm') and hasattr(model.tts_model.base_lm, 'kv_cache'):
                cache_wrapper = model.tts_model.base_lm.kv_cache
                if hasattr(cache_wrapper, 'kv_cache') and isinstance(cache_wrapper.kv_cache, torch.Tensor):
                    if cache_wrapper.kv_cache.device != device:
                        cache_wrapper.kv_cache = cache_wrapper.kv_cache.to(device)
                        logger.info(f"Moved tts_model.base_lm.kv_cache.kv_cache (Tensor) to {device}")
            if hasattr(model.tts_model, 'residual_lm') and hasattr(model.tts_model.residual_lm, 'kv_cache'):
                cache_wrapper = model.tts_model.residual_lm.kv_cache
                if hasattr(cache_wrapper, 'kv_cache') and isinstance(cache_wrapper.kv_cache, torch.Tensor):
                     if cache_wrapper.kv_cache.device != device:
                        cache_wrapper.kv_cache = cache_wrapper.kv_cache.to(device)
                        logger.info(f"Moved tts_model.residual_lm.kv_cache.kv_cache (Tensor) to {device}")
        except Exception as e:
            logger.warning(f"Could not move KV Caches: {e}")

        # 3. 移动 Denoiser (尝试移动整个 Denoiser)
        if hasattr(model, 'denoiser') and isinstance(model.denoiser, torch.nn.Module):
            model.denoiser.to(device)
            logger.info(f"Moved denoiser (nn.Module) to {device}")
        elif hasattr(model, 'denoiser') and hasattr(model.denoiser, '_pipeline') and isinstance(model.denoiser._pipeline, torch.nn.Module):
            model.denoiser._pipeline.to(device)
            logger.info(f"Moved denoiser._pipeline (nn.Module) to {device}")
        elif hasattr(model, 'denoiser') and hasattr(model.denoiser, '_pipeline') and hasattr(model.denoiser._pipeline, 'model') and isinstance(model.denoiser._pipeline.model, torch.nn.Module):
            model.denoiser._pipeline.model.to(device)
            logger.info(f"Moved denoiser._pipeline.model (nn.Module) to {device}")
        
        # 4. 检查 VAE/Vocoder
        if hasattr(model, 'vocoder') and isinstance(model.vocoder, torch.nn.Module):
             model.vocoder.to(device)
             logger.info(f"Moved vocoder (nn.Module) to {device}")
             
        # 5. 检查 TextNormalizer
        if hasattr(model, 'text_normalizer') and isinstance(model.text_normalizer, torch.nn.Module):
            model.text_normalizer.to(device)
            logger.info(f"Moved text_normalizer (nn.Module) to {device}")

        logger.info(f"--- Finished moving VoxCPM components to {device} ---")
    except Exception as e:
        logger.error(f"Error during move_voxcpm_to_device: {e}")
        
def offload_voxcpm(model, offload_device):
    """将 VoxCPM 的所有已知 torch.nn.Module 和缓存卸载到 CPU"""
    logger.info(f"--- Offloading VoxCPM components to {offload_device} ---")
    try:
        # 1. 卸载 TTS Model
        if hasattr(model, 'tts_model') and isinstance(model.tts_model, torch.nn.Module):
            model.tts_model.to(offload_device)
            logger.info(f"Offloaded tts_model (nn.Module) to {offload_device}")

        # 2. 卸载 TTS Model 的 KV 缓存
        try:
            if hasattr(model.tts_model, 'base_lm') and hasattr(model.tts_model.base_lm, 'kv_cache'):
                cache_wrapper = model.tts_model.base_lm.kv_cache
                if hasattr(cache_wrapper, 'kv_cache') and isinstance(cache_wrapper.kv_cache, torch.Tensor):
                    if cache_wrapper.kv_cache.device != offload_device:
                        cache_wrapper.kv_cache = cache_wrapper.kv_cache.to(offload_device)
                        logger.info(f"Offloaded tts_model.base_lm.kv_cache.kv_cache (Tensor) to {offload_device}")
            if hasattr(model.tts_model, 'residual_lm') and hasattr(model.tts_model.residual_lm, 'kv_cache'):
                cache_wrapper = model.tts_model.residual_lm.kv_cache
                if hasattr(cache_wrapper, 'kv_cache') and isinstance(cache_wrapper.kv_cache, torch.Tensor):
                    if cache_wrapper.kv_cache.device != offload_device:
                        cache_wrapper.kv_cache = cache_wrapper.kv_cache.to(offload_device)
                        logger.info(f"Offloaded tts_model.residual_lm.kv_cache.kv_cache (Tensor) to {offload_device}")
        except Exception as e:
            logger.warning(f"Could not offload KV Caches: {e}")

        # 3. 卸载 Denoiser (与加载逻辑一致)
        if hasattr(model, 'denoiser') and isinstance(model.denoiser, torch.nn.Module):
            model.denoiser.to(offload_device)
            logger.info(f"Offloaded denoiser (nn.Module) to {offload_device}")
        elif hasattr(model, 'denoiser') and hasattr(model.denoiser, '_pipeline') and isinstance(model.denoiser._pipeline, torch.nn.Module):
            model.denoiser._pipeline.to(offload_device)
            logger.info(f"Offloaded denoiser._pipeline (nn.Module) to {offload_device}")
        elif hasattr(model, 'denoiser') and hasattr(model.denoiser, '_pipeline') and hasattr(model.denoiser._pipeline, 'model') and isinstance(model.denoiser._pipeline.model, torch.nn.Module):
            model.denoiser._pipeline.model.to(offload_device)
            logger.info(f"Offloaded denoiser._pipeline.model (nn.Module) to {offload_device}")

        # 4. 卸载 VAE/Vocoder
        if hasattr(model, 'vocoder') and isinstance(model.vocoder, torch.nn.Module):
             model.vocoder.to(offload_device)
             logger.info(f"Offloaded vocoder (nn.Module) to {offload_device}")

        # 5. 卸载 TextNormalizer
        if hasattr(model, 'text_normalizer') and isinstance(model.text_normalizer, torch.nn.Module):
            model.text_normalizer.to(offload_device)
            logger.info(f"Offloaded text_normalizer (nn.Module) to {offload_device}")
             
        logger.info(f"--- Finished offloading VoxCPM components to {offload_device} ---")
    except Exception as e:
        logger.error(f"Error during offload_voxcpm: {e}")


# --- (辅助函数：get_tts_model_folders, parse_time, parse_srt 保持不变) ---
def get_tts_model_folders():
    try:
        tts_root = folder_paths.get_folder_paths("tts")[0]
    except (KeyError, IndexError):
        logger.error("TTS directory not registered! Please check __init__.py.")
        return []
    model_folders = []
    if os.path.isdir(tts_root):
        for d in os.listdir(tts_root):
            if os.path.isdir(os.path.join(tts_root, d)):
                model_folders.append(d)
    return sorted(model_folders)

def parse_time(ts: str) -> float:
    try:
        h, m, s_rest = ts.split(':')
        s, ms = s_rest.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except ValueError:
        logger.warning(f"Failed to parse timestamp: {ts}")
        return 0.0

def parse_srt(srt_text: str, is_multi_speaker_mode: bool):
    entries = []
    speaker_regex = re.compile(r"^(\S+)\s+(.*)") 
    blocks = srt_text.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            index_str = lines[0].strip()
            time_line = lines[1]
            start_ts, end_ts = time_line.split(' --> ')
            start_sec = parse_time(start_ts.strip())
            end_sec = parse_time(end_ts.strip())
            raw_text = " ".join(lines[2:]).strip()
            speaker_name = "default"
            clean_text = raw_text
            if is_multi_speaker_mode:
                match = speaker_regex.match(raw_text)
                if match:
                    speaker_name = match.group(1).strip()
                    clean_text = match.group(2).strip()
                else:
                    speaker_name = "default"
                    clean_text = raw_text
            else:
                speaker_name = "default"
                clean_text = raw_text
            if clean_text:
                entries.append((index_str, start_sec, end_sec, speaker_name, clean_text))
        except Exception as e:
            logger.warning(f"Skipping malformed SRT block: {block}. Error: {e}")
    return entries

# --- (节点 0, 1, 2 保持不变) ---
class VoxCPM_Loader:
    CATEGORY = "audio/tts"
    RETURN_TYPES = ("VOXCPM_MODEL",)
    FUNCTION = "load_model"
    @classmethod
    def INPUT_TYPES(s):
        model_names = get_tts_model_folders()
        model_names.insert(0, "openbmb/VoxCPM-0.5B (Auto-Download)")
        return {
            "required": {
                "model_name": (model_names, {"default": model_names[0]}),
                "optimize": (["none", "no_fullgraph", "fullgraph"], {"default": "none"}), 
                "load_denoiser": ("BOOLEAN", {"default": True}),
            }
        }
    @classmethod
    def IS_CHANGED(s, model_name, optimize, load_denoiser): 
        return f"{model_name}-{optimize}-{load_denoiser}"
    def load_model(self, model_name, optimize, load_denoiser):
        hf_id = "openbmb/VoxCPM-0.5B"
        try:
            tts_root = folder_paths.get_folder_paths("tts")[0]
        except (KeyError, IndexError):
            models_dir = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
            tts_root = os.path.join(models_dir, "TTS")
            os.makedirs(tts_root, exist_ok=True)
        load_path = ""
        download_cache_dir = None
        if model_name == f"{hf_id} (Auto-Download)":
            load_path = hf_id
            download_cache_dir = tts_root
            logger.info(f"Auto-downloading '{hf_id}' to '{tts_root}'...")
        else:
            load_path = os.path.join(tts_root, model_name)
            if not os.path.isdir(load_path):
                logger.error(f"Could not find local model folder: {load_path}")
                logger.warning(f"Falling back to auto-download: {hf_id}")
                load_path = hf_id
                download_cache_dir = tts_root
        logger.info(f"Loading VoxCPM model from: {load_path}...")
        try:
            model = voxcpm.VoxCPM.from_pretrained(
                hf_model_id=load_path, 
                load_denoiser=load_denoiser,
                cache_dir=download_cache_dir,
                optimization_mode=optimize 
            )
            return (model,)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}'. Error: {e}")
            raise e

class VoxCPM_Cache_Builder:
    CATEGORY = "audio/tts"
    RETURN_TYPES = ("VOXCPM_CACHE",)
    FUNCTION = "build_cache"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VOXCPM_MODEL", ), 
                "speaker_name": ("STRING", {"default": "speaker1"}),
                "prompt_audio": ("AUDIO", ),
                "prompt_text": ("STRING", {"multiline": True, "default": "Enter prompt transcript..."}),
            }
        }
    def build_cache(self, model, speaker_name, prompt_audio, prompt_text):
        if not prompt_text or prompt_text == "Enter prompt transcript...":
            raise ValueError("Prompt text is required for voice cloning.")
        if model is None:
            raise RuntimeError("Model not loaded. Please connect a VoxCPM_Loader node.")
        device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        waveform = prompt_audio['waveform']
        sample_rate = prompt_audio['sample_rate']
        temp_path = ""
        try:
            logger.info(f"Moving VoxCPM model to {device} for Cache Building...")
            move_voxcpm_to_device(model, device)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                temp_path = tmp_file.name
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)
            sf.write(temp_path, waveform.T.cpu().numpy(), sample_rate)
            logger.info(f"Building cache for speaker: '{speaker_name}'")
            start_time = time.time()
            cache = model.tts_model.build_prompt_cache(
                prompt_wav_path=temp_path,
                prompt_text=prompt_text
            )
            logger.info(f"Cache for '{speaker_name}' built in {time.time() - start_time:.2f}s")
            cache_object = {
                "speaker_name": speaker_name,
                "cache": cache
            }
            return (cache_object,)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if model is not None:
                logger.info(f"Offloading VoxCPM model from Cache Builder to {offload_device}...")
                offload_voxcpm(model, offload_device)
                model_management.soft_empty_cache()

class VoxCPM_Cache_Combiner:
    CATEGORY = "audio/tts"
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "cache_item": ("VOXCPM_CACHE",), }, "optional": { "cache_group": ("CACHE_GROUP",) } }
    RETURN_TYPES = ("CACHE_GROUP",)
    FUNCTION = "combine_caches"
    def combine_caches(self, cache_item, cache_group=None):
        if cache_group is None:
            cache_group = {}
        else:
            cache_group = cache_group.copy()
        speaker_name = cache_item.get("speaker_name")
        cache = cache_item.get("cache")
        if speaker_name and cache:
            if speaker_name in cache_group:
                logger.warning(f"Duplicate speaker name '{speaker_name}' in Cache Combiner. Overwriting.")
            cache_group[speaker_name] = cache
        logger.info(f"Combined caches. Current speakers: {list(cache_group.keys())}")
        return (cache_group,)


# --- 节点 3: SRT 处理器 (已清理) ---
class VoxCPM_SRT_Processor:
    CATEGORY = "audio/tts"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process_srt"

    @classmethod
    def INPUT_TYPES(s):
        stretch_options = ["none"]
        if LIBROSA_AVAILABLE:
            stretch_options.append("librosa")
        if PYDUB_AVAILABLE:
            stretch_options.append("pydub")
        return {
            "required": {
                "model": ("VOXCPM_MODEL", ), 
                "cache_group": ("CACHE_GROUP", ),
                "srt_text": ("STRING", {"multiline": True, "default": "1\n00:00:00,500 --> 00:00:02,000\nspeaker1 Hello world.\n"}),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "stretch_method": (stretch_options, {"default": "librosa"}),
                "stretch_n_fft": ("INT", {"default": 320, "min": 128, "max": 8192, "step": 8}),
                "stretch_hop_length": ("INT", {"default": 8, "min": 8, "max": 2048, "step": 8}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "inference_timesteps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "retry_max_attempts": ("INT", {"default": 3, "min": 0, "max": 10}),
                "retry_threshold": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 20.0}),
            }
        }
    
    def process_srt(self, model, cache_group, srt_text, normalize_text,
                    stretch_method, 
                    stretch_n_fft, stretch_hop_length,
                    keep_model_loaded,
                    cfg_value, inference_timesteps, seed, retry_max_attempts, retry_threshold):

        if model is None:
            raise RuntimeError("Model not loaded. Please connect a VoxCPM_Loader node.")

        device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        try:
            logger.info(f"Moving VoxCPM model to {device} for SRT Processing...")
            move_voxcpm_to_device(model, device)
            
            if seed == -1:
                seed = torch.randint(0, 0x7FFFFFFFFFFFFFFF, (1,)).item()
            torch.manual_seed(seed)
            if normalize_text and model.text_normalizer is None:
                logger.info("Initializing Text Normalizer...")
                model.text_normalizer = text_normalize.TextNormalizer()
            single_speaker_name = None
            if len(cache_group) == 1:
                single_speaker_name = list(cache_group.keys())[0]
                logger.info(f"Single speaker mode detected. Assigning all text to: '{single_speaker_name}'")
            parsed_entries = parse_srt(srt_text, is_multi_speaker_mode=(single_speaker_name is None))
            if not parsed_entries:
                raise ValueError("SRT text could not be parsed or is empty.")
            SAMPLE_RATE = 16000 
            total_duration_sec = parsed_entries[-1][2]
            total_samples = int((total_duration_sec + 1.0) * SAMPLE_RATE)
            timeline_audio = np.zeros(total_samples, dtype=np.float32)
            logger.info(f"Created audio timeline of {total_duration_sec:.2f}s")
            for i, (index_str, start_sec, end_sec, speaker_name, clean_text) in enumerate(parsed_entries):
                if single_speaker_name:
                    speaker_name = single_speaker_name
                if not single_speaker_name and speaker_name == "default":
                    logger.warning(f"Line '{clean_text[:30]}...' has no speaker tag in multi-speaker mode. Skipping.")
                    continue
                if speaker_name not in cache_group: 
                    logger.warning(f"Speaker '{speaker_name}' in SRT not found in Cache Group. Skipping.")
                    continue
                if normalize_text:
                    clean_text = model.text_normalizer.normalize(clean_text)
                logger.info(f"Generating for: [{start_sec:.2f}s] {speaker_name}: '{clean_text[:50]}...'")
                cache_to_use = cache_group[speaker_name]
                gen_result = model.tts_model.generate_with_prompt_cache(
                    target_text=clean_text,
                    prompt_cache=cache_to_use,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
                (wav_tensor, new_tokens, new_feats) = gen_result 
                wav = wav_tensor.squeeze().cpu().numpy()
                if stretch_method != "none":
                    audio_duration_sec = len(wav) / SAMPLE_RATE
                    srt_duration_sec = end_sec - start_sec
                    if audio_duration_sec > srt_duration_sec and srt_duration_sec > 0.1:
                        stretch_rate = audio_duration_sec / srt_duration_sec
                        logger.info(f"    Stretching audio: {audio_duration_sec:.2f}s -> {srt_duration_sec:.2f}s (Rate: {stretch_rate:.2f}x)")
                        if stretch_method == "pydub":
                            if not PYDUB_AVAILABLE:
                                raise ImportError("pydub is required for 'pydub' stretch method. Please run 'pip install pydub'.")
                            logger.info("    Using pydub (High Quality)...")
                            try:
                                wav_int16 = (wav * 32767).astype(np.int16)
                                audio_segment = pydub.AudioSegment(
                                    data=wav_int16.tobytes(),
                                    sample_width=2, 
                                    frame_rate=SAMPLE_RATE,
                                    channels=1
                                )
                                new_audio = audio_segment.speedup(playback_speed=stretch_rate)
                                wav_int16_new = np.frombuffer(new_audio.raw_data, dtype=np.int16)
                                wav = wav_int16_new.astype(np.float32) / 32768.0
                            except pydub.exceptions.CouldntEncodeError as e:
                                logger.error(f"pydub/ffmpeg failed! Is ffmpeg installed and in your system's PATH? Error: {e}")
                                raise e
                        elif stretch_method == "librosa":
                            if not LIBROSA_AVAILABLE:
                                raise ImportError("Librosa is required for 'librosa' stretch method. Please run 'pip install librosa'.")
                            logger.info("    Using librosa (Fallback Quality)...")
                            logger.info(f"    Using stretch params: n_fft={stretch_n_fft}, hop_length={stretch_hop_length}")
                            wav = librosa.effects.time_stretch(
                                wav, 
                                rate=stretch_rate,
                                n_fft=stretch_n_fft,
                                hop_length=stretch_hop_length
                            )
                        target_samples = int(srt_duration_sec * SAMPLE_RATE)
                        if len(wav) > target_samples:
                            logger.info(f"    Truncating stretched audio to fit target {target_samples} samples.")
                            wav = wav[:target_samples]
                start_sample = int(start_sec * SAMPLE_RATE)
                end_sample = start_sample + len(wav)
                if end_sample > len(timeline_audio):
                    wav_to_place = wav[:len(timeline_audio) - start_sample]
                else:
                    wav_to_place = wav
                timeline_audio[start_sample : start_sample + len(wav_to_place)] += wav_to_place
            
            final_waveform = torch.from_numpy(timeline_audio).float().unsqueeze(0)
            final_audio = {"waveform": final_waveform.unsqueeze(0), "sample_rate": SAMPLE_RATE}
            
            return (final_audio,)

        finally:
            if not keep_model_loaded:
                logger.info(f"Unloading VoxCPM model to {offload_device}...")
                offload_voxcpm(model, offload_device)
                logger.info("Clearing VRAM cache...")
                model_management.soft_empty_cache()


# --- 节点 4: SRT 配音器 (已清理) ---
class VoxCPM_SRT_Dubber:
    CATEGORY = "audio/tts"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process_dub"

    @classmethod
    def INPUT_TYPES(s):
        stretch_options = ["none"]
        if LIBROSA_AVAILABLE:
            stretch_options.append("librosa")
        if PYDUB_AVAILABLE:
            stretch_options.append("pydub")
        return {
            "required": {
                "model": ("VOXCPM_MODEL", ), 
                "cache_group": ("CACHE_GROUP", ),
                "original_audio": ("AUDIO", ),
                "srt_text": ("STRING", {"multiline": True}),
                "entries_to_replace": ("STRING", {"default": "1 2"}),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "stretch_method": (stretch_options, {"default": "librosa"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "stretch_n_fft": ("INT", {"default": 320, "min": 128, "max": 8192, "step": 8}),
                "stretch_hop_length": ("INT", {"default": 8, "min": 8, "max": 2048, "step": 8}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "inference_timesteps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "retry_max_attempts": ("INT", {"default": 3, "min": 0, "max": 10}),
                "retry_threshold": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 20.0}),
            }
        }
    
    def process_dub(self, model, cache_group, original_audio, srt_text, entries_to_replace,
                    normalize_text, stretch_method, 
                    keep_model_loaded,
                    stretch_n_fft, stretch_hop_length,
                    cfg_value, inference_timesteps, seed, retry_max_attempts, retry_threshold):

        if model is None:
            raise RuntimeError("Model not loaded. Please connect a VoxCPM_Loader node.")

        device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        try:
            logger.info(f"Moving VoxCPM model to {device} for SRT Dubbing...")
            move_voxcpm_to_device(model, device)
            
            if seed == -1:
                seed = torch.randint(0, 0x7FFFFFFFFFFFFFFF, (1,)).item()
            torch.manual_seed(seed)
            if normalize_text and model.text_normalizer is None:
                logger.info("Initializing Text Normalizer...")
                model.text_normalizer = text_normalize.TextNormalizer()
            VOX_SR = 16000
            ORIG_SR = original_audio["sample_rate"]
            waveform_tensor = original_audio["waveform"].squeeze(0)
            if waveform_tensor.dim() == 2 and waveform_tensor.shape[0] > 1:
                logger.info(f"Original audio is stereo, converting to mono by averaging channels.")
                waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)
            timeline_audio_np = waveform_tensor.squeeze().clone().cpu().numpy()
            replace_set = set(entries_to_replace.split())
            single_speaker_name = None
            if len(cache_group) == 1:
                single_speaker_name = list(cache_group.keys())[0]
                logger.info(f"Single speaker mode detected. Assigning all text to: '{single_speaker_name}'")
            parsed_entries = parse_srt(srt_text, is_multi_speaker_mode=(single_speaker_name is None))
            if not parsed_entries:
                raise ValueError("SRT text could not be parsed or is empty.")
            logger.info(f"Starting dubbing process. Will replace entries: {replace_set}")
            for i, (index_str, start_sec, end_sec, speaker_name, clean_text) in enumerate(parsed_entries):
                if single_speaker_name:
                    speaker_name = single_speaker_name
                if index_str not in replace_set:
                    continue
                if not single_speaker_name and speaker_name == "default":
                    logger.warning(f"Line '{clean_text[:30]}...' (entry {index_str}) has no speaker tag. Skipping replacement.")
                    continue
                if speaker_name not in cache_group: 
                    logger.warning(f"Speaker '{speaker_name}' for entry {index_str} not found in Cache Group. Skipping replacement.")
                    continue
                if normalize_text:
                    clean_text = model.text_normalizer.normalize(clean_text)
                logger.info(f"Replacing entry {index_str}: [{start_sec:.2f}s] {speaker_name}: '{clean_text[:50]}...'")
                cache_to_use = cache_group[speaker_name]
                gen_result = model.tts_model.generate_with_prompt_cache(
                    target_text=clean_text,
                    prompt_cache=cache_to_use,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
                (wav_tensor, new_tokens, new_feats) = gen_result 
                wav_16k = wav_tensor.squeeze().cpu().numpy()
                srt_duration_sec = end_sec - start_sec
                if stretch_method != "none":
                    audio_duration_sec = len(wav_16k) / VOX_SR
                    if audio_duration_sec > srt_duration_sec and srt_duration_sec > 0.1:
                        stretch_rate = audio_duration_sec / srt_duration_sec
                        logger.info(f"    Stretching audio: {audio_duration_sec:.2f}s -> {srt_duration_sec:.2f}s (Rate: {stretch_rate:.2f}x)")
                        if stretch_method == "pydub" and PYDUB_AVAILABLE:
                            logger.info("    Using pydub (High Quality)...")
                            try:
                                wav_int16 = (wav_16k * 32767).astype(np.int16)
                                audio_segment = pydub.AudioSegment(data=wav_int16.tobytes(), sample_width=2, frame_rate=VOX_SR, channels=1)
                                new_audio = audio_segment.speedup(playback_speed=stretch_rate)
                                wav_int16_new = np.frombuffer(new_audio.raw_data, dtype=np.int16)
                                wav_16k = wav_int16_new.astype(np.float32) / 32768.0
                            except pydub.exceptions.CouldntEncodeError as e:
                                logger.error(f"pydub/ffmpeg failed! Is ffmpeg installed? Error: {e}")
                                raise e
                        elif stretch_method == "librosa" and LIBROSA_AVAILABLE:
                            logger.info("    Using librosa (Fallback Quality)...")
                            wav_16k = librosa.effects.time_stretch(wav_16k, rate=stretch_rate, n_fft=stretch_n_fft, hop_length=stretch_hop_length)
                        target_samples_16k = int(srt_duration_sec * VOX_SR)
                        if len(wav_16k) > target_samples_16k:
                            wav_16k = wav_16k[:target_samples_16k]
                if ORIG_SR != VOX_SR:
                    logger.info(f"    Resampling generated audio from {VOX_SR}Hz to {ORIG_SR}Hz...")
                    if not LIBROSA_AVAILABLE:
                         raise ImportError("Librosa is required for resampling. Please run 'pip install librosa'.")
                    wav_orig_sr = librosa.resample(wav_16k, orig_sr=VOX_SR, target_sr=ORIG_SR)
                else:
                    wav_orig_sr = wav_16k
                wav = wav_orig_sr
                start_sample = int(start_sec * ORIG_SR)
                end_sample = int(end_sec * ORIG_SR)
                if end_sample > len(timeline_audio_np):
                    logger.warning(f"Entry {index_str}: SRT end time {end_sec}s exceeds original audio length. Extending timeline.")
                    padding = np.zeros(end_sample - len(timeline_audio_np))
                    timeline_audio_np = np.concatenate([timeline_audio_np, padding])
                if start_sample > len(timeline_audio_np):
                     logger.error(f"Entry {index_str}: Start sample {start_sample} is after end of audio. Skipping.")
                     continue
                logger.info(f"    Muting original audio from {start_sample} to {end_sample}")
                timeline_audio_np[start_sample : end_sample] = 0.0
                new_end_sample = start_sample + len(wav)
                if new_end_sample > len(timeline_audio_np):
                    wav_to_place = wav[:len(timeline_audio_np) - start_sample]
                else:
                    wav_to_place = wav
                logger.info(f"    Inserting new audio ({len(wav_to_place)} samples) at {start_sample}")
                timeline_audio_np[start_sample : start_sample + len(wav_to_place)] += wav_to_place

            final_waveform = torch.from_numpy(timeline_audio_np).float().unsqueeze(0)
            final_audio = {"waveform": final_waveform.unsqueeze(0), "sample_rate": ORIG_SR}
            
            return (final_audio,)

        finally:
            if not keep_model_loaded:
                logger.info(f"Unloading VoxCPM model to {offload_device}...")
                offload_voxcpm(model, offload_device)
                logger.info("Clearing VRAM cache...")
                model_management.soft_empty_cache()


# --- (节点 5 和 MAPPINGS 保持不变) ---
class Audio_Trimmer_By_Timestamp:
    CATEGORY = "audio/tts"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "trim_audio"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ), 
                "timestamp": ("STRING", {"multiline": False, "default": "00:00:00,000 --> 00:00:05,000"}),
            }
        }
    def trim_audio(self, audio, timestamp):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if "-->" not in timestamp:
            logger.error("Timestamp format invalid. Expected 'HH:MM:SS,mmm --> HH:MM:SS,mmm'.")
            return (audio,)
        try:
            start_ts_str, end_ts_str = timestamp.split(' --> ')
            start_sec = parse_time(start_ts_str.strip())
            end_sec = parse_time(end_ts_str.strip())
            if start_sec >= end_sec:
                logger.error(f"Start time ({start_sec}s) is after end time ({end_sec}s). Returning original audio.")
                return (audio,)
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            total_samples = waveform.shape[2]
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            if start_sample >= end_sample:
                logger.error("Calculated audio segment is empty. Returning original audio.")
                return (audio,)
            trimmed_waveform = waveform[:, :, start_sample:end_sample]
            logger.info(f"Audio trimmed: {start_sec:.3f}s to {end_sec:.3f}s. New shape: {trimmed_waveform.shape}")
            trimmed_audio = {
                "waveform": trimmed_waveform,
                "sample_rate": sample_rate
            }
            return (trimmed_audio,)
        except Exception as e:
            logger.error(f"Failed to parse timestamp '{timestamp}'. Error: {e}")
            return (audio,)

# ... (保留你原来的所有代码，将以下内容追加到文件末尾) ...

# --- 节点 6: SRT 自动配音器 (逐句参考源音频) ---
class VoxCPM_SRT_Auto_Dubber:
    CATEGORY = "audio/tts"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process_auto_dub"

    @classmethod
    def INPUT_TYPES(s):
        stretch_options = ["none"]
        if LIBROSA_AVAILABLE:
            stretch_options.append("librosa")
        if PYDUB_AVAILABLE:
            stretch_options.append("pydub")
        return {
            "required": {
                "model": ("VOXCPM_MODEL", ), 
                "original_audio": ("AUDIO", ),
                "source_srt_text": ("STRING", {"multiline": True, "default": "Source Language SRT (Transcript)..."}),
                "target_srt_text": ("STRING", {"multiline": True, "default": "Target Language SRT (Translation)..."}),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "stretch_method": (stretch_options, {"default": "librosa"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "stretch_n_fft": ("INT", {"default": 320, "min": 128, "max": 8192, "step": 8}),
                "stretch_hop_length": ("INT", {"default": 8, "min": 8, "max": 2048, "step": 8}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "inference_timesteps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "retry_max_attempts": ("INT", {"default": 3, "min": 0, "max": 10}),
                "retry_threshold": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 20.0}),
            }
        }
    
    def process_auto_dub(self, model, original_audio, source_srt_text, target_srt_text,
                         normalize_text, stretch_method, 
                         keep_model_loaded,
                         stretch_n_fft, stretch_hop_length,
                         cfg_value, inference_timesteps, seed, retry_max_attempts, retry_threshold):

        if model is None:
            raise RuntimeError("Model not loaded. Please connect a VoxCPM_Loader node.")

        device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        # 解析两个字幕文件
        logger.info("Parsing Source and Target SRTs...")
        source_entries = parse_srt(source_srt_text, is_multi_speaker_mode=False)
        target_entries = parse_srt(target_srt_text, is_multi_speaker_mode=False)

        # 将源字幕转换为字典，以索引(Index)为Key，方便查找
        # source_dict结构: {'1': (start, end, text), '2': ...}
        source_dict = {entry[0]: entry for entry in source_entries}

        try:
            logger.info(f"Moving VoxCPM model to {device} for Auto-Dubbing...")
            move_voxcpm_to_device(model, device)
            
            if seed == -1:
                seed = torch.randint(0, 0x7FFFFFFFFFFFFFFF, (1,)).item()
            torch.manual_seed(seed)
            
            if normalize_text and model.text_normalizer is None:
                logger.info("Initializing Text Normalizer...")
                model.text_normalizer = text_normalize.TextNormalizer()

            # 准备音频数据
            VOX_SR = 16000
            ORIG_SR = original_audio["sample_rate"]
            waveform_tensor = original_audio["waveform"].squeeze(0) # [channels, samples]
            
            # 确保处理的是单声道用于切片，但保留原始声道信息可能用于最终混合（这里简单起见，时间轴使用单声道合成，后续可扩展）
            if waveform_tensor.dim() == 2 and waveform_tensor.shape[0] > 1:
                logger.info(f"Original audio is stereo, converting to mono for processing context.")
                waveform_mono = torch.mean(waveform_tensor, dim=0)
            else:
                waveform_mono = waveform_tensor.squeeze()
            
            # 准备输出的时间轴
            # 我们以原始音频长度为基准，复制一份全静音的或者保留原始背景音（这里逻辑是Dubbing，通常保留背景音很难分离，所以我们假设完全替换对应片段）
            # 简单起见，创建一个全静音的画布，长度等于原音频。
            # *如果用户希望保留背景音，需要在ComfyUI里做Audio Mix，本节点只输出人声*
            timeline_audio_np = np.zeros(waveform_mono.shape[0], dtype=np.float32)
            
            logger.info(f"Starting Auto-Dubbing process. Found {len(target_entries)} target lines.")

            for i, (t_index, t_start, t_end, _, t_text) in enumerate(target_entries):
                # 1. 在源字幕中找到对应行（用于提取参考音频和参考文本）
                if t_index not in source_dict:
                    logger.warning(f"Line index {t_index} found in Target SRT but missing in Source SRT. Skipping.")
                    continue
                
                s_index, s_start, s_end, _, s_text = source_dict[t_index]
                
                # 2. 切割源音频作为 Reference
                # 计算采样点
                s_start_sample = int(s_start * ORIG_SR)
                s_end_sample = int(s_end * ORIG_SR)
                
                # 边界检查
                s_start_sample = max(0, s_start_sample)
                s_end_sample = min(len(waveform_mono), s_end_sample)
                
                if s_end_sample - s_start_sample < 1600: # 小于0.1秒忽略
                    logger.warning(f"Source audio segment {t_index} too short. Skipping.")
                    continue

                ref_wav_slice = waveform_mono[s_start_sample:s_end_sample]
                
                # 3. 构建临时的 Prompt Cache (这是本节点的灵魂：每一句都现场学习)
                temp_wav_path = ""
                prompt_cache = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        temp_wav_path = tmp_file.name
                    
                    # 写入临时文件供VoxCPM读取
                    sf.write(temp_wav_path, ref_wav_slice.cpu().numpy(), ORIG_SR)
                    
                    # 归一化 Prompt Text (源文本)
                    clean_prompt_text = s_text
                    if normalize_text:
                        clean_prompt_text = model.text_normalizer.normalize(clean_prompt_text)

                    # 构建 Cache
                    # logger.info(f"Building cache for Line {t_index} using source audio ({s_start:.1f}-{s_end:.1f}s)")
                    prompt_cache = model.tts_model.build_prompt_cache(
                        prompt_wav_path=temp_wav_path,
                        prompt_text=clean_prompt_text
                    )
                except Exception as e:
                    logger.error(f"Failed to build cache for line {t_index}: {e}")
                    continue
                finally:
                    if temp_wav_path and os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)

                # 4. 生成目标音频
                clean_target_text = t_text
                if normalize_text:
                    clean_target_text = model.text_normalizer.normalize(clean_target_text)
                
                logger.info(f"Generating Line {t_index}: '{clean_target_text[:30]}...'")
                
                gen_result = model.tts_model.generate_with_prompt_cache(
                    target_text=clean_target_text,
                    prompt_cache=prompt_cache, # 使用刚才现场生成的Cache
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
                
                (wav_tensor, _, _) = gen_result
                wav_16k = wav_tensor.squeeze().cpu().numpy()
                
                # 5. 后处理：变速与对齐 (对齐到 Target SRT 的时间轴)
                target_duration = t_end - t_start
                
                # 如果需要变速
                if stretch_method != "none":
                    current_duration = len(wav_16k) / VOX_SR
                    if current_duration > target_duration and target_duration > 0.1:
                        stretch_rate = current_duration / target_duration
                        # 限制最大变速比，避免过分鬼畜
                        stretch_rate = min(stretch_rate, 2.0) 
                        
                        if stretch_method == "pydub" and PYDUB_AVAILABLE:
                            try:
                                wav_int16 = (wav_16k * 32767).astype(np.int16)
                                audio_segment = pydub.AudioSegment(data=wav_int16.tobytes(), sample_width=2, frame_rate=VOX_SR, channels=1)
                                new_audio = audio_segment.speedup(playback_speed=stretch_rate)
                                wav_int16_new = np.frombuffer(new_audio.raw_data, dtype=np.int16)
                                wav_16k = wav_int16_new.astype(np.float32) / 32768.0
                            except Exception:
                                pass # Fallback handled naturally or just skip stretch
                        elif stretch_method == "librosa" and LIBROSA_AVAILABLE:
                            wav_16k = librosa.effects.time_stretch(wav_16k, rate=stretch_rate, n_fft=stretch_n_fft, hop_length=stretch_hop_length)
                        
                        # 硬截断以防止溢出
                        max_samples = int(target_duration * VOX_SR)
                        if len(wav_16k) > max_samples:
                            wav_16k = wav_16k[:max_samples]

                # 6. 重采样回原始采样率
                if ORIG_SR != VOX_SR and LIBROSA_AVAILABLE:
                    wav_final = librosa.resample(wav_16k, orig_sr=VOX_SR, target_sr=ORIG_SR)
                else:
                    wav_final = wav_16k

                # 7. 放入时间轴
                t_start_sample = int(t_start * ORIG_SR)
                t_end_sample = t_start_sample + len(wav_final)
                
                # 扩展时间轴如果不够长
                if t_end_sample > len(timeline_audio_np):
                    padding = np.zeros(t_end_sample - len(timeline_audio_np), dtype=np.float32)
                    timeline_audio_np = np.concatenate([timeline_audio_np, padding])
                
                # 叠加（这里选择直接覆盖，因为是配音；如果重叠则相加）
                # 简单的淡入淡出可以防止爆音，这里先做简单覆盖
                timeline_audio_np[t_start_sample : t_end_sample] = wav_final

            # 封装输出
            final_waveform = torch.from_numpy(timeline_audio_np).float().unsqueeze(0) # [1, samples]
            final_audio = {"waveform": final_waveform.unsqueeze(0), "sample_rate": ORIG_SR} # [1, 1, samples]
            
            return (final_audio,)

        finally:
            if not keep_model_loaded:
                logger.info(f"Unloading VoxCPM model to {offload_device}...")
                offload_voxcpm(model, offload_device)
                model_management.soft_empty_cache()


# --- 节点 7: 文本文件加载器 (处理路径和引号) ---
class Load_Text_From_File:
    CATEGORY = "utils"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_file"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "C:\\path\\to\\subtitle.srt"}),
            }
        }

    def load_file(self, file_path):
        # 1. 清理路径：移除首尾的引号 (Windows "复制为路径" 经常带引号)
        cleaned_path = file_path.strip().strip('"').strip("'")
        
        # 2. 检查文件是否存在
        if not os.path.exists(cleaned_path):
            logger.error(f"File not found: {cleaned_path}")
            return (f"Error: File not found at {cleaned_path}",)
        
        # 3. 尝试读取内容
        try:
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return (content,)
        except UnicodeDecodeError:
            # 尝试 fallback 编码
            try:
                with open(cleaned_path, 'r', encoding='gbk') as f:
                    content = f.read()
                return (content,)
            except Exception as e:
                 return (f"Error reading file: {e}",)
        except Exception as e:
            return (f"Error reading file: {e}",)


# --- 更新映射表 ---

NODE_CLASS_MAPPINGS = {
    "VoxCPM_Loader": VoxCPM_Loader,
    "VoxCPM_Cache_Builder": VoxCPM_Cache_Builder,
    "VoxCPM_Cache_Combiner": VoxCPM_Cache_Combiner,
    "VoxCPM_SRT_Processor": VoxCPM_SRT_Processor,
    "VoxCPM_SRT_Dubber": VoxCPM_SRT_Dubber,
    "VoxCPM_SRT_Auto_Dubber": VoxCPM_SRT_Auto_Dubber, # 新增
    "Audio_Trimmer_By_Timestamp": Audio_Trimmer_By_Timestamp,
    "Load_Text_From_File": Load_Text_From_File,       # 新增
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxCPM_Loader": "VoxCPM Loader",
    "VoxCPM_Cache_Builder": "VoxCPM Cache Builder",
    "VoxCPM_Cache_Combiner": "VoxCPM Cache Combiner (Chainable)",
    "VoxCPM_SRT_Processor": "VoxCPM SRT Processor (from Scratch)",
    "VoxCPM_SRT_Dubber": "VoxCPM SRT Dubber (Replace Audio)",
    "VoxCPM_SRT_Auto_Dubber": "VoxCPM SRT Auto-Dubber (Line-by-Line Ref)", # 新增
    "Audio_Trimmer_By_Timestamp": "Audio Trimmer (by Timestamp)",
    "Load_Text_From_File": "Load Text from File (Path)",                   # 新增
}