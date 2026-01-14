# inference.py
import torch
import librosa
import soundfile as sf
from pathlib import Path
from model import load_wakeword_model
from manage_audio import AudioPreprocessor  # your preprocessor

class WakeWordDetector:
    """Inference pipeline for wake word detection"""
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_length = 101
        self.preprocessor = AudioPreprocessor(
            sr=16000,
            n_dct_filters=40,
            n_mels=40,
            f_max=4000,
            f_min=20,
            n_fft=480,
            hop_ms=10
        )
        self.model = load_wakeword_model(model_path, target_length=self.target_length, device=self.device)

    def convert_audio_to_wav(self, audio_path, target_sr=16000):
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() == '.wav':
            try:
                y, sr = librosa.load(str(audio_path), sr=None)
                if sr == target_sr:
                    return str(audio_path)
            except:
                pass
        y, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        temp_wav = audio_path.parent / f"{audio_path.stem}_converted.wav"
        sf.write(str(temp_wav), y, target_sr)
        return str(temp_wav)

    def preprocess_audio(self, audio_path):
        wav_path = self.convert_audio_to_wav(audio_path)
        mfccs = self.preprocessor.compute_mfccs(wav_path)
        audio_tensor = torch.FloatTensor(mfccs)
        if audio_tensor.shape[0] < self.target_length:
            padding = torch.zeros(self.target_length - audio_tensor.shape[0], audio_tensor.shape[1], audio_tensor.shape[2])
            audio_tensor = torch.cat([audio_tensor, padding], dim=0)
        else:
            audio_tensor = audio_tensor[:self.target_length]
        audio_tensor = audio_tensor.permute(2, 0, 1).unsqueeze(0)
        return audio_tensor.to(self.device)

    def predict(self, audio_path, return_confidence=True):
        tensor = self.preprocess_audio(audio_path)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        if return_confidence:
            return pred.item(), conf.item(), probs.cpu().numpy()[0]
        else:
            return pred.item()

    def predict_with_details(self, audio_path, threshold=0.5):
        pred, conf, probs = self.predict(audio_path)
        detected = (pred == 1) and (probs[1] >= threshold)
        return {
            'audio_file': str(audio_path),
            'wake_word_detected': detected,
            'prediction': 'Wake Word' if pred == 1 else 'Not Wake Word',
            'confidence': f"{conf*100:.2f}%",
            'probabilities': {'negative': f"{probs[0]*100:.2f}%", 'positive': f"{probs[1]*100:.2f}%"},
            'threshold_used': threshold
        }
