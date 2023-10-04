import whisper
import torch
import numpy as np
import subprocess
from subprocess import CalledProcessError


def SegmentAudio(_model_path_or_name, _device, _audio_path_or_name):
    print("Load Segment Audio")
    audios = []
    list_audio_30secs = LoadAudioCustom(_audio_path_or_name, 16000, 25)
    for audio in list_audio_30secs:
        audios.append(whisper.pad_or_trim(audio))
    model = whisper.load_model(_model_path_or_name, device=_device)
    results = ""
    i = 1
    for chunk in audios:
        print("chunk# " + str(i))
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        # decode the audio

        options = whisper.DecodingOptions(
            language="en", without_timestamps=False, fp16=False
        )
        result = whisper.decode(model, mel, options)
        results += result.text
        i = i + 1

    # print the recognized text
    print(results)


def LoadAudioCustom(file: str, sr: int = 16000, segment_length_secs: int = 30):
    # run ffmpeg
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]

    try:
        out = subprocess.check_output(cmd)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # create numpy array
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # Split audio into segments of specified length
    segment_length_samples = sr * segment_length_secs
    segments = [
        audio_data[i : i + segment_length_samples]
        for i in range(0, len(audio_data), segment_length_samples)
    ]

    return segments


def FullAudio(_model_path_or_name, _device, _audio_path_or_name):
    print("Load Full Audio")
    model = whisper.load_model(_model_path_or_name, device=_device)
    _decode_options = {
        "without_timestamps": True,
        "language": "en",
        "fp16": False,
    }
    print("Transcribing...")
    result = model.transcribe(_audio_path_or_name, **_decode_options)
    text = result["text"]
    print(text)


def Main(_model_path_or_name, device, audio_path_or_name, _segmentaudio=False):
    if _segmentaudio:
        SegmentAudio(
            _model_path_or_name,
            device,
            audio_path_or_name,
        )
    else:
        FullAudio(_model_path_or_name, device, audio_path_or_name)


if __name__ == "__main__":
    torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path_or_name = "fables_01_00_aesop.mp3"
    model_path_or_name = "base"
    print("device: " + device)
    print("filename or path: " + audio_path_or_name)
    Main(model_path_or_name, device, audio_path_or_name)
