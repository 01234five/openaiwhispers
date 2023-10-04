import whisper
import torch
import numpy as np
import subprocess
from subprocess import CalledProcessError
def main(_device,_audio_path_or_name):
    audios = []
    list_audio_30secs = load_audio_custom(_audio_path_or_name)
    for audio in list_audio_30secs:
        audios.append(whisper.pad_or_trim(audio))
    model = whisper.load_model("base", device=_device)    
    results = ""
    i=1
    for chunk in audios:
        print("chunk# "+str(i))
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        # decode the audio

        options = whisper.DecodingOptions(language="en", without_timestamps=False, fp16 = False)
        result = whisper.decode(model, mel, options)    
        results += result.text
        i=i+1

    # print the recognized text
    print(results)


def load_audio_custom(file: str, sr: int = 16000, segment_length_secs: int = 30):
    #run ffmpeg
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]

    try:
        out = subprocess.check_output(cmd)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    #create numpy array
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # Split audio into segments of specified length
    segment_length_samples = sr * segment_length_secs
    segments = [audio_data[i:i + segment_length_samples] for i in range(0, len(audio_data), segment_length_samples)]

    return segments


if __name__ == "__main__":
    torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path_or_name="fables_01_00_aesop.mp3"
    print("device: "+device)
    print("filename or path: "+ audio_path_or_name)
    main(device,audio_path_or_name)