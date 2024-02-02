Project Title: Real-Time Drowsiness Detection System
=====

## Project Overview:

This project develops a Real-Time Drowsiness Detection System aimed at enhancing road safety by identifying signs of driver fatigue. Utilizing advanced machine learning models, the system detects and alerts drivers when symptoms of microsleep are observed.

## Our Detection System:

The system operates in real-time, analyzing visual cues from the driver to identify states of alertness, microsleep, and yawning. Key features include:

- Real-time drowsiness detection using live camera feed.
- Alerts for detected microsleep instances.
- Integration with Google Maps API to suggest nearby rest areas.

## Dataset Exploration:
The dataset can be found here: [FL3D Dataset](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d)
- FL3D was constructed using the NITYMED dataset (Night-Time Yawning-Microsleep-Eyeblink-driver Distraction)
- FL3D takes frames from NITYMED and labels each frame with one of the following three labels: [alert, microsleep. yawning].
- FL3D dataset consists of 53331 images. 35483 alert images, 8074 microsleep images and 4770 yawning images.

## Main Technologies:
### Streamlit (UI):
For building a robust and interactive web application for real-time inference, we have used Streamlit and used Embedded architecture to load the model directly into the UI.

#### SignIn/SignUp page:

#### Detection Page:

#### Nearest Location Page:

## Installation and Setup:

#### 1. Clone the Repository:
`git clone git@github.com:Safwan-ullah-khan/dsp-finalproject.git`

#### 2. Create conda environment (Optional but Recommended):
`conda create --name myenv python==3.9`

#### 3. Install Required Python Libraries:
`pip install -r requirements.txt`

#### 4. Run the Streamlit App:
`streamlit run main.py
`