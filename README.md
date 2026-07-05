# Human Scream Detection and Analysis for Crime

This project aims to develop a Machine Learning (ML) model capable of detecting human screams for help from audio input. By analyzing various types of voices, including human screams, the model determines whether the input audio contains a scream for help or not. This can be a useful tool for crime prevention and emergency detection.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

## Features
- Detects human screams for help from audio input.
- Trained on diverse audio datasets to distinguish between screams and other sounds.
- Supports majority voting ensemble of multiple ML models for robust predictions.
- Provides an easy-to-use web interface for uploading audio files and receiving predictions.
- Outputs clear results ("Yes" or "No") indicating the presence of a scream for help.

## Technologies Used
- **Python**: Core programming language for development.
- **Flask**: Backend framework for building the web application.
- **Librosa**: For audio processing and feature extraction.
- **Machine Learning Models**:
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN) Classifier
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier
- **Joblib**: For saving and loading ML models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/man180705/Human-Scream-Detection.git
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the pre-trained model files (`decision_tree_classifier.pkl`, `knn_classifier.pkl`, etc.) in the project directory.

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
    http://127.0.0.1:5000/
   ```

3. Upload an audio file through the provided interface.

4. The application will process the file and display the result: 
   - `Result: Yes` if the audio contains a scream for help.
   - `Result: No` otherwise.

## Dataset
The model is trained on a curated dataset containing:
- Human screams in various emotional states (e.g., fear, panic).
- Non-scream human voices (e.g., talking, laughing).
- Environmental and background noises.

**Note**: The dataset used is not included in this repository due to size and privacy constraints.

## Model Architecture

The system uses an ensemble learning approach with the following steps:
1. **Audio Feature Extraction**: Extracts MFCCs, spectral contrast, chroma, and other audio features using Librosa.
2. **Model Predictions**: Uses multiple ML models to predict the class of the input audio.
3. **Majority Voting**: Aggregates predictions from all models to determine the final result.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.
