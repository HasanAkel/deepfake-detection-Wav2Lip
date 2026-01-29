%%writefile /content/Wav2Lip/sync_checker_final.py
import os
import cv2
import torch
import librosa
import numpy as np
import subprocess
import mediapipe as mp
import traceback
from models import SyncNet_color

print("=== FINAL SYNC CHECKER WITH FACE DETECTION ===")

# ==========================================
# AUDIO
# ==========================================
def get_mel(audio_path):
    wav, sr = librosa.load(audio_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=800,
        hop_length=200,
        win_length=800,
        n_mels=80,
        fmin=55,
        fmax=7600
    )

    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - 20 + 100) / 100, 0, 1) * 2 - 1
    return mel


# ==========================================
# FACE + MOUTH DETECTOR
# ==========================================
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

MOUTH_LANDMARKS = list(range(61, 88))  # lips


def extract_mouth(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    xs = [int(landmarks[i].x * w) for i in MOUTH_LANDMARKS]
    ys = [int(landmarks[i].y * h) for i in MOUTH_LANDMARKS]

    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    mouth = frame[y1:y2, x1:x2]
    mouth = cv2.resize(mouth, (96, 96))
    return mouth


# ==========================================
# PREPROCESSOR
# ==========================================
class Preprocessor:
    def generate(self, video_path, audio_path):
        cap = cv2.VideoCapture(video_path)
        mel = get_mel(audio_path)

        fps = 25
        mel_fps = 80
        buffer = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mouth = extract_mouth(frame)
            if mouth is None:
              buffer.clear()   # break temporal window
              idx += 1
              continue


            buffer.append(mouth)

            if len(buffer) >= 5:
                center = idx - 2
                mel_idx = int(center * mel_fps / fps)

                start = mel_idx - 8
                end = mel_idx + 8

                if start >= 0 and end < mel.shape[1]:
                    vid = np.array(buffer[-5:])
                    vid = np.transpose(vid, (3, 0, 1, 2))
                    vid = vid.reshape(15, 96, 96)
                    vid = torch.FloatTensor(vid).unsqueeze(0) / 255.0

                    aud = torch.FloatTensor(mel[:, start:end]).unsqueeze(0).unsqueeze(0)

                    yield vid, aud

            idx += 1

        cap.release()


# ==========================================
# SYNC DETECTOR
# ==========================================
class SyncDetector:
    def __init__(self, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SyncNet_color().to(self.device)

        ckpt = torch.load(weights, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}

        self.model.load_state_dict(state)
        self.model.eval()

    def distance(self, aud, vid):
        with torch.no_grad():
            aud, vid = aud.to(self.device), vid.to(self.device)
            a, v = self.model(aud, vid)
            assert a.shape == v.shape == (1, 512)
            return torch.norm(a - v, p=2).item()


# ==========================================
# ANALYSIS
# ==========================================
def analyze(video, detector, prep):
    print(f"\nAnalyzing: {os.path.basename(video)}")

    aud = "temp.wav"
    vid = "temp.mp4"

    subprocess.call(f'ffmpeg -y -i "{video}" -ar 16000 "{aud}" -loglevel error', shell=True)
    subprocess.call(f'ffmpeg -y -i "{video}" -r 25 "{vid}" -loglevel error', shell=True)

    dists = []

    try:
        for i, (v, a) in enumerate(prep.generate(vid, aud)):
            if i % 3 == 0:
                dists.append(detector.distance(a, v))
    except Exception:
        traceback.print_exc()

    for f in [aud, vid]:
        if os.path.exists(f):
            os.remove(f)

    if not dists:
        return None

    return {
        "mean": np.mean(dists),
        "std": np.std(dists),
        "max": np.max(dists),
        "p95": np.percentile(dists, 95)
    }


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    VIDEO_DIR = "sample_data"
    WEIGHTS = "checkpoints/lipsync_expert.pth"

    detector = SyncDetector(WEIGHTS)
    prep = Preprocessor()

    for f in os.listdir(VIDEO_DIR):
        if f.endswith((".mp4", ".avi", ".mov")):
            stats = analyze(os.path.join(VIDEO_DIR, f), detector, prep)
            print(f, stats)
