import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#AUDIO_PATH = "data/audio/let_it_be.wav"
AUDIO_PATH = "data/audio/I_ll_Follow_the_Sun.wav"

def plot_waveform(y, sr):
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform (music)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    #plt.savefig("waveform_2.png")
    

def plot_chroma(chroma, sr, hop_length):
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(
        chroma,
        x_axis="time",
        y_axis="chroma",
        sr=sr,
        hop_length=hop_length,
        cmap="magma",
    )
    plt.title("Chroma CQT (music)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    #plt.savefig("chroma_2.png")
    
def main():
    print("Loading audio...")
    y, sr = librosa.load(AUDIO_PATH, sr=22050)  # Downsample to 22kHz

    print(f"Audio loaded: {AUDIO_PATH}")
    print(f"Duration: {len(y)/sr:.2f} seconds")

    plot_waveform(y, sr)

    print("Applying HPSS separation...")
    y_harm, y_perc = librosa.effects.hpss(y)

    print("Extracting chroma features...")
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)

    print("Chroma shape:", chroma.shape)

    plot_chroma(chroma, sr, hop_length)

    print("DONE ✔ Preprocessing successful.")


if __name__ == "__main__":
    main()