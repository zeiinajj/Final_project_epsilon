import ast
import warnings
from io import BytesIO

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydub import AudioSegment
from scipy import stats

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

genres = pd.read_csv("genres.csv")
id_to_title = dict(zip(genres["genre_id"], genres["title"]))


def columns():
    feature_sizes = dict(
        chroma_cens=12,
        chroma_cqt=12,
        chroma_stft=12,
        mfcc=20,
        rmse=1,
        spectral_bandwidth=1,
        spectral_centroid=1,
        spectral_contrast=7,
        spectral_rolloff=1,
        tonnetz=6,
        zcr=1,
    )
    moments = ("mean", "std", "skew", "kurtosis", "median", "min", "max")

    columns = []
    for feature, size in feature_sizes.items():
        total = size * len(moments)
        for i in range(total):
            if i == 0:
                columns.append(feature)
            else:
                columns.append(f"{feature}.{i}")

    return columns


def compute_features(audio_object: BytesIO):
    features = pd.Series(index=columns(), dtype=np.float32, name="input_audio")
    warnings.filterwarnings("error", module="librosa")

    def feature_stats(name, values):
        stats_list = [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            stats.skew(values, axis=1),
            stats.kurtosis(values, axis=1),
            np.median(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
        ]

        idx = 0
        for s in stats_list:
            for v in s:
                if idx == 0:
                    features[f"{name}"] = v
                else:
                    features[f"{name}.{idx}"] = v
                idx += 1

    try:
        audio_object.seek(0)
        audio = AudioSegment.from_file(audio_object, format="mp3")

        # Step 3: Convert pydub audio to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Step 4: Normalize and reshape if stereo
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)  # Convert to mono

        samples /= np.iinfo(audio.array_type).max  # Normalize to [-1.0, 1.0]

        # Step 5: (Optional) Resample using librosa
        sr = audio.frame_rate
        x = librosa.resample(samples, orig_sr=sr, target_sr=sr)

        print("X SHAPE:")
        print(x.shape, np.isnan(x).any())

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats("zcr", f)

        cqt = np.abs(
            librosa.cqt(
                x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None
            )
        )
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats("chroma_cqt", f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats("chroma_cens", f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats("tonnetz", f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        print("STFT SHAPE:")
        print(stft.shape)  # Should be (freq_bins, time_frames)

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats("chroma_stft", f)

        f = librosa.feature.rms(S=stft)
        feature_stats("rmse", f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats("spectral_centroid", f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats("spectral_bandwidth", f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats("spectral_contrast", f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats("spectral_rolloff", f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        print("Audio shape:", x.shape)
        print("STFT shape:", stft.shape)
        print("RMS:", librosa.feature.rms(S=stft).shape)
        print(
            "MFCC:", librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20).shape
        )
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats("mfcc", f)

    except Exception as e:
        st.error(f"Error processing audio: {e}")

    return features


# streamlit

st.set_page_config(page_title="Music Genre Classifier", layout="wide")
st.title(" Genre Classifier ")

tab1, tab2 = st.tabs(["Visual Stats", " Audio file classification"])

with tab1:

    df = pd.read_csv("dataset.csv")
    genres = pd.read_csv("genres.csv")

    df["genres"] = df["genres"].apply(
        lambda x: (
            [int(i) for i in ast.literal_eval(x)]
            if isinstance(x, str)
            else [int(i) for i in x] if isinstance(x, list) else []
        )
    )

    id_to_title = dict(zip(genres["genre_id"], genres["title"]))

    # Map IDs to titles
    df["genres_named"] = df["genres"].apply(
        lambda id_list: (
            [id_to_title.get(i) for i in id_list] if isinstance(id_list, list) else []
        )
    )

    st.header("Visualized Statistics")
    # Histogram of genres
    st.subheader("Genre Distribution")
    fig_genre = px.histogram(data_frame=df, x="genres_named", text_auto=True)
    st.plotly_chart(fig_genre)

    # Selected features to visualize
    selected_features = ["rmse.1", "spectral_centroid.1", "spectral_bandwidth.1"]

    # Histograms for selected features
    st.subheader("Histograms of Selected Features")
    for feature in selected_features:
        fig = px.histogram(df, x=feature, nbins=50, title=f"Distribution of {feature}")
        st.plotly_chart(fig)

    # Boxplots for selected features
    st.subheader("Boxplots of Selected Features")
    for feature in selected_features:
        fig = px.box(df, y=feature, title=f"Boxplot of {feature}")
        st.plotly_chart(fig)

    # Pairplot (Scatter matrix) of selected features
    st.subheader("Pairplot (Scatter Matrix)")
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    fig = px.scatter_matrix(
        sample_df,
        dimensions=selected_features,
        title="Pairplot of Selected Features (Sample of 1000)",
    )
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)

with tab2:

    st.header("Upload an audio file for genre classification")

    uploaded_audio = st.file_uploader("Upload an audio file (MP3)", type=["mp3", "wav"])

    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/mp3")

        st.info("Extracting features and predicting genre...")
        features = compute_features(uploaded_audio)
        print(features)

        X_input = features.values.reshape(1, -1)
        X_input_scaled = scaler.transform(X_input)

        predicted_genre_id = model.predict(X_input_scaled)[0]

        genre_name = id_to_title.get(predicted_genre_id, "Unknown Genre")

        st.success(f"Predicted Genre: **{genre_name}**")
