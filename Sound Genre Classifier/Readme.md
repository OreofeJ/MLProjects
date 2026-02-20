# Sound Genre Classifier System

## Overview
This project develops and compares different machine learning models for music genre classification based on audio features and spectrogram images. The goal is to accurately categorize music tracks into one of ten genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using both traditional audio feature extraction and deep learning techniques.

## Dataset
The project utilizes a dataset containing various audio features and spectrogram images derived from music tracks. The main feature dataset (`Datasetfilename.csv`) includes:
- **Filename**: Original audio file name.
- **Length**: Duration of the audio track.
- **Chroma STFT**: Chromagram from a Short-Time Fourier Transform (mean and variance).
- **RMS**: Root Mean Square energy (mean and variance).
- **Spectral Centroid**: Centroid of the spectrum (mean and variance).
- **Spectral Bandwidth**: Bandwidth of the spectrum (mean and variance).
- **Rolloff**: Roll-off frequency (mean and variance).
- **Zero Crossing Rate**: Rate of sign-changes along a signal (mean and variance).
- **Harmony**: Harmonic content (mean and variance).
- **Perceptr**: Perceptual features (mean and variance).
- **Tempo**: Estimated tempo in beats per minute (BPM).
- **MFCCs (1-20)**: Mel-frequency cepstral coefficients (mean and variance for each of 20 coefficients).
- **Label**: The music genre (target variable).

An image dataset of spectrograms (located at `DatasetImagePath`) is used for the Convolutional Neural Network (CNN) models.

## Exploratory Data Analysis (EDA)
- **Data Structure**: The dataset consists of 1000 entries with 60 features, including spectral, rhythmic, and MFCC features, along with the `label` as the target.
- **Missing Values**: No missing values were found in the dataset.
- **Feature Distributions**: Many features, such as MFCCs, exhibit skewed distributions, indicating the need for scaling prior to model training.
- **Correlation**: A heatmap revealed significant correlations among MFCC features and spectral features, suggesting multicollinearity, which might impact linear models.
- **Genre Overlap**: Visualizations (e.g., box plots of Spectral Centroid by Genre, scatter plots of MFCC1 vs. Spectral Centroid for Jazz vs. Metal) showed some overlap in feature ranges across different genres, highlighting the challenge of classification.

## Models Implemented

### 1. XGBoost Classifier (with Handcrafted Audio Features)
- **Input**: Scaled numerical audio features (MFCCs, spectral features, etc.).
- **Preprocessing**: `LabelEncoder` for target variable, `StandardScaler` for features.
- **Architecture**: `XGBClassifier` with `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="multi:softprob"`, `num_class=10`, `eval_metric="mlogloss"`.
- **Evaluation**: Standard classification metrics (accuracy, precision, recall, F1-score) and a confusion matrix.

### 2. Convolutional Neural Network (CNN) from Scratch (with Spectrogram Images)
- **Input**: Spectrogram images (resized to 128x128 pixels).
- **Preprocessing**: `ImageDataGenerator` for rescaling and validation split.
- **Architecture**: Sequential model with multiple `Conv2D`, `BatchNormalization`, `MaxPooling2D` layers, followed by `Flatten`, `Dense` (256 units with ReLU, Dropout 0.5), and a final `Dense` (10 units with Softmax).
- **Compilation**: `optimizer="adam"`, `loss="categorical_crossentropy"`, `metrics=["accuracy"]`.
- **Training**: 15 epochs with `ImageDataGenerator`.

### 3. Transfer Learning CNN (with EfficientNetB0 and Spectrogram Images)
- **Input**: Spectrogram images (resized to 224x224 pixels).
- **Preprocessing**: `ImageDataGenerator` for rescaling and validation split.
- **Architecture**: `EfficientNetB0` (pretrained on ImageNet, `include_top=False`, `trainable=False`) as the base model, followed by `GlobalAveragePooling2D`, `Dense` (256 units with ReLU, BatchNormalization, Dropout 0.5), and a final `Dense` (10 units with Softmax).
- **Compilation**: `tf.keras.optimizers.Adam(learning_rate=1e-3)`, `loss="categorical_crossentropy"`, `metrics=["accuracy"]`.
- **Training**: Initial training for 20 epochs with `EarlyStopping` and `ReduceLROnPlateau` callbacks.
- **Fine-tuning**: Unfroze the top layers of `EfficientNetB0` (`base_model.trainable = True`, `for layer in base_model.layers[:-20]: layer.trainable = False`) and recompiled with a lower learning rate (`tf.keras.optimizers.Adam(learning_rate=1e-4)`), then trained for 10 additional epochs.

## Key Findings
- **XGBoost Performance**: The XGBoost model, utilizing handcrafted audio features, achieved a high accuracy of **91.44%** on the test set. This demonstrates the effectiveness of traditional machine learning with well-engineered features for this task.
- **CNN (From Scratch) Performance**: The custom CNN model performed poorly, with a maximum validation accuracy of approximately **12.06%**.
- **Transfer Learning CNN Performance**: The Transfer Learning CNN with EfficientNetB0 also struggled significantly, achieving a final validation accuracy of around **10.05%**.
- **CNN Underperformance Analysis**: Diagnostic checks revealed that both CNN models consistently predicted only one class for all samples in the validation set, indicating a complete failure to learn meaningful patterns. This suggests issues such as:
    - **Limited Image Dataset Size**: The number of spectrogram images available for training CNNs might be insufficient.
    - **Low-Quality Spectrograms**: The pre-rendered spectrograms might not capture sufficient discriminatory information or have inconsistencies.
    - **Domain Mismatch**: The ImageNet pretraining of EfficientNetB0 might not be directly transferable or beneficial for audio spectrogram classification without substantial fine-tuning on a larger, domain-specific image dataset.

## Conclusion

The CNN models underperformed due to the limited size of the image dataset, low-quality pre-rendered spectrograms, and the mismatch between ImageNet pretraining and audio-domain representations. In contrast, XGBoost achieved high accuracy by leveraging robust, handcrafted audio features specifically designed for music genre discrimination.


This highlights the importance of appropriate feature representation (handcrafted vs. raw spectrograms) and sufficient, high-quality data for deep learning approaches, especially when dealing with domain-specific tasks like audio analysis.
