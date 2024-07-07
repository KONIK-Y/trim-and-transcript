import os
import shutil
import sys
import argparse

import pandas as pd
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydantic import BaseModel
from typing import Optional, List
import torch


class Segments(BaseModel):
    path: str
    start: int
    end: int
    transcript: Optional[str] = None


# default parameters
minutes = 10
silence_threshold = -50

preprocessed_dir = "./assets/preprocessed"
outputs_dir = "./assets/outputs"
model = whisper.load_model(
    name="base",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
should_clean_up = False


def get_input_audio() -> AudioSegment:
    file_path = input("対象の音声ファイルを選択してください: ")
    if not os.path.exists(file_path):
        raise FileNotFoundError("ファイルが見つかりません。")

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".wav":
        audio = AudioSegment.from_wav(file_path)
    elif file_extension == ".mp3":
        audio = AudioSegment.from_mp3(file_path)
    elif file_extension == ".m4a":
        audio = AudioSegment.from_file(file_path, format="m4a")
    elif file_extension == ".mp4":
        audio = AudioSegment.from_file(file_path, format="mp4")
    else:
        raise ValueError("サポートされていない音声形式です。")
    return audio


def save_audio(segments: AudioSegment, filename: str) -> str:
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    save_path = os.path.join(preprocessed_dir, filename)
    segments.export(save_path, format="wav")
    return save_path


def trim_audio(audio: AudioSegment, min_length: int, silence_thresh: int) -> List[Segments]:
    ms = min_length * 60 * 1000
    chunks = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=silence_thresh)

    results = []
    checkpoint = 0
    for start, end in chunks:
        if end - checkpoint >= ms:
            segment = audio[checkpoint:end]
            file_path = save_audio(segment, f"segment_{checkpoint}-{end}.wav")
            results.append(Segments(path=file_path, start=checkpoint, end=end))
            checkpoint = end

    if checkpoint < (audio.duration_seconds * 1000):
        file_path = save_audio(audio[checkpoint:], f"segment_{checkpoint}-last.wav")
        results.append(Segments(path=file_path, start=checkpoint, end=int(audio.duration_seconds * 1000)))
    return results


def generate_transcript(path: str) -> str:
    with torch.no_grad():
        transcript = model.transcribe(
            path,
            verbose=True,
            beam_size=5,
            fp16=True,
            without_timestamps=False,
        )
    return transcript["text"]


def main():
    audio = get_input_audio()
    targets = trim_audio(audio, minutes, silence_threshold)
    for i, segment in enumerate(targets):
        print(f"processing segment {i + 1}: {segment.start}-{segment.end}")
        if segment.transcript is None:
            segment.transcript = generate_transcript(segment.path)

    df = pd.DataFrame(
        {
            "start": [segment.start for segment in targets],
            "end": [segment.end for segment in targets],
            "path": [segment.path for segment in targets],
            "transcript": [segment.transcript for segment in targets],
        },
        columns=["start", "end", "path", "transcript"],
    )
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    df.to_csv(f"{outputs_dir}/output.csv", index=False)


def clean_up():
    if os.path.exists(preprocessed_dir):
        shutil.rmtree(preprocessed_dir)


if __name__ == "__main__":
    main()
    if should_clean_up:
        clean_up()
