"""
ZipEnhancer Module - Audio Denoising Enhancer

Provides on-demand import ZipEnhancer functionality for audio denoising processing.
Related dependencies are imported only when denoising functionality is needed.
"""

import os
import tempfile
from typing import Optional, Union
import soundfile as sf
import torchaudio
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer"""
    def __init__(self, 
                 model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
                 device: str = "auto" 
                 ):
        """
        Initialize ZipEnhancer
        Args:
            model_path: ModelScope model path or local path
            device: Device to run on ('auto', 'cuda', 'cpu') # <-- 添加注释
        """
        self.model_path = model_path

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        print(f"[ZipEnhancer] Initializing on device: {device}")

        self._pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=self.model_path,
                device=device
            )
        
    def _normalize_loudness(self, wav_path: str):
        """
        Audio loudness normalization
        
        Args:
            wav_path: Audio file path
        """
        # --- 修复: 使用 soundfile 绕过 torchcodec ---
        try:
            # 注意: 变量是 wav_path
            audio_data, sr = sf.read(wav_path, dtype='float32')
        except Exception as e:
            print(f"[VoxCPM-Local-Fix] soundfile.read failed: {e}")
            raise e

        if audio_data.ndim == 2:
            audio_data = audio_data.T
        else:
            audio_data = audio_data[None, :]

        audio = torch.from_numpy(audio_data)
        # --- 修复结束 ---
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20-loudness)
        torchaudio.save(wav_path, normalized_audio, sr)
    
    def enhance(self, input_path: str, output_path: Optional[str] = None, 
                normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If pipeline is not initialized or processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
        try:
            # Perform denoising processing
            self._pipeline(input_path, output_path=output_path)
            # Loudness normalization
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as e:
            # Clean up possibly created temporary files
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")