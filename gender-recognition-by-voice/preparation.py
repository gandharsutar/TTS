import glob
import os
import pandas as pd
import numpy as np
import shutil
import librosa
from scipy.signal import spectrogram
from scipy.fft import dct
from glob import glob
from tqdm import tqdm


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)

    # Function to convert Hz to Mel
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    # Function to convert Mel to Hz
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    # Function to create Mel filter bank
    def get_mel_filter_bank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
        if fmax is None:
            fmax = sr / 2
        
        # Create Mel points
        mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert Hz points to FFT bin numbers
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(n_mels):
            # Create triangular filters
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            
            for i in range(left, center):
                filter_bank[m, i] = (i - left) / (center - left)
            for i in range(center, right):
                filter_bank[m, i] = (right - i) / (right - center)
        return filter_bank

    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        # SciPy based Mel Spectrogram
        # Parameters for spectrogram
        nperseg = 2048  # Window size
        noverlap = nperseg // 2  # Overlap
        
        # Compute spectrogram
        frequencies, times, Sxx = spectrogram(X, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Convert to power spectrogram (magnitude squared)
        power_spectrogram = np.abs(Sxx)**2
        
        # Create Mel filter bank
        n_fft = nperseg
        n_mels = 128  # Standard for librosa melspectrogram
        mel_filter_bank = get_mel_filter_bank(sample_rate, n_fft, n_mels=n_mels)
        
        # Apply Mel filter bank to power spectrogram
        mel_spectrogram = np.dot(mel_filter_bank, power_spectrogram)
        
        # Apply log scale (similar to librosa's default)
        mel = np.mean(librosa.power_to_db(mel_spectrogram).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

dirname = "data"

if not os.path.isdir(dirname):
    os.mkdir(dirname)


csv_files = glob.glob("*.csv")

for j, csv_file in enumerate(csv_files):
    print("[+] Preprocessing", csv_file)
    df = pd.read_csv(csv_file)
    # only take filename and gender columns
    new_df = df[["filename", "gender"]]
    print("Previously:", len(new_df), "rows")
    # take only male & female genders (i.e droping NaNs & 'other' gender)
    new_df = new_df[np.logical_or(new_df['gender'] == 'female', new_df['gender'] == 'male')]
    print("Now:", len(new_df), "rows")
    new_csv_file = os.path.join(dirname, csv_file)
    # save new preprocessed CSV 
    new_df.to_csv(new_csv_file, index=False)
    # get the folder name
    folder_name, _ = csv_file.split(".")
    audio_files = glob.glob(f"{folder_name}/{folder_name}/*")
    all_audio_filenames = set(new_df["filename"])
    for i, audio_file in tqdm(list(enumerate(audio_files)), f"Extracting features of {folder_name}"):
        splited = os.path.split(audio_file)
        # audio_filename = os.path.join(os.path.split(splited[0])[-1], splited[-1])
        audio_filename = f"{os.path.split(splited[0])[-1]}/{splited[-1]}"
        # print("audio_filename:", audio_filename)
        if audio_filename in all_audio_filenames:
            # print("Copyying", audio_filename, "...")
            src_path = f"{folder_name}/{audio_filename}"
            target_path = f"{dirname}/{audio_filename}"
            #create that folder if it doesn't exist
            if not os.path.isdir(os.path.dirname(target_path)):
                os.mkdir(os.path.dirname(target_path))
            features = extract_feature(src_path, mel=True)
            target_filename = target_path.split(".")[0]
            np.save(target_filename, features)
            # shutil.copyfile(src_path, target_path)