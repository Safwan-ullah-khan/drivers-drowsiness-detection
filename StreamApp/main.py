import streamlit as st
import cv2
from mtcnn import MTCNN
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import pygame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from db_ops import save_user_to_db, validate_user, history

model = load_model("../../model/model_res_test.h5")
label_mapping = {'alert': 0, 'microsleep': 1, 'yawning': 2}

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'auth'

if 'detection_started' not in st.session_state:
    st.session_state.detection_started = False


def main():
    if st.session_state.current_page == 'auth':
        auth_page()
    elif st.session_state.logged_in:
        app_menu()


def auth_page():
    st.title("Welcome to Drowsiness Detection App")
    col1, col2 = st.columns([1, 2])

    col1.image("alertnap.jpeg", width=200)
    col2.subheader("User Authentication")

    auth_option = col2.radio("Choose an option", ["Sign In", "Sign Up"])

    if auth_option == "Sign In":
        sign_in_page()
    elif auth_option == "Sign Up":
        sign_up_page()


def app_menu():
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    menu_options = ["Detection", "Nearest Rest Areas", "History", "Logout"]
    selected_option = st.sidebar.radio("Select an option", menu_options)

    if selected_option == "Detection":
        detection_page()
    elif selected_option == "Nearest Rest Areas":
        open_test_map_page()
    elif selected_option == "History":
        show_history()
    elif selected_option == "Logout":
        st.session_state.logged_in = False
        st.session_state.current_page = 'auth'
        st.experimental_rerun()


def detection_page():
    st.title("Real-time Detection")
    st.markdown(
        """
        Welcome to the Drowsiness Detection App! This app uses real-time facial recognition to detect signs of drowsiness 
        such as microsleep, yawning, and alert states. Click the 'Toggle Detection' button to begin or stop monitoring.
        """
    )
    # Set background image
    st.image("background.jpeg", use_column_width=True)
    toggle_button = st.button("Start 🚀/ Stop detection 🛑", key="toggle_button")

    if toggle_button:
        st.session_state.detection_started = not st.session_state.detection_started

        if st.session_state.detection_started:
            start_detection()


microsleep_counter = 0


def start_detection():
    global microsleep_counter
    video_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    while st.session_state.detection_started:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            (x, y, w, h) = face['box']
            face_frame = frame[y:y + h, x:x + w]

            face_frame = cv2.resize(face_frame, (128, 128))
            face_frame = face_frame.astype('float32')
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)

            prediction = model.predict(face_frame).argmax(axis=-1)
            state_label = [
                label for label, index in label_mapping.items() if
                index == prediction
            ][0]
            # save_prediction_to_db(state_label)
            if state_label == 'microsleep':
                microsleep_counter += 1
                if microsleep_counter >= 3:
                    show_alert()
                    cap.release()
                    st.session_state.detection_started = False
            else:
                microsleep_counter = 0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'State: {state_label}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the OpenCV frame to a format that can be displayed in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the video frame using the st.image function
        video_placeholder.image(frame_rgb, channels="RGB",
                                use_column_width=True)


def play_sound(sound_file):
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load the sound file
    pygame.mixer.music.load(sound_file)
    # Play the sound
    pygame.mixer.music.play()
    # Wait for the sound to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def show_alert():
    st.warning("Warning: Microsleep detected multiple times!")
    st.write("ALERT SOUND PLAYING")
    sound_file = "./alert.wav"
    play_sound(sound_file)

    # Display a message to the user
    st.warning("You seem tired. Consider taking a break and finding a rest area.")
    st.info("Locating nearest rest areas...")

    # Open nearest rest areas map page
    open_test_map_page()


def stop_sound():
    if st.button("Get Location"):
        # if get_location:
        open_test_map_page()
    pygame.mixer.music.stop()


def stop_detection():
    pass


def show_history():
    st.write("Prediction History Page")
    history_data = history()

    if history_data:
        df = pd.DataFrame(history_data, columns=["Timestamp", "State"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Extract the hour from the timestamp
        df["Hour"] = df["Timestamp"].dt.hour

        # Filter for the three previous dates
        date_range = pd.date_range(end=pd.Timestamp("now"), periods=3, freq="D")
        df_filtered = df[df["Timestamp"].dt.date.isin(date_range.date)]

        # Count occurrences of each state for each hour
        count_df = pd.crosstab(index=df_filtered["Hour"], columns=[df_filtered["Timestamp"].dt.date, df_filtered["State"]])

        # Create subplots for each date
        plt.figure(figsize=(15, 15))
        for i, date in enumerate(date_range.date):
            plt.subplot(3, 1, i+1)
            sns.lineplot(data=count_df[date], markers=True, palette={"microsleep": "red", "alert": "green", "yawning": "black"})
            plt.title(f"State Evolution by Hour ({date})")
            plt.xlabel("Hour of the Day")
            plt.ylabel("State Count")
            plt.legend(title="State")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plot using Streamlit
        st.pyplot(plt.gcf())
    else:
        st.write("No history data available.")


def open_test_map_page():
    with open("../Map/testMap.html", "r") as html_file:
        html_content = html_file.read()
    # Embed the HTML content
    st.components.v1.html(html_content, height=1000)


def sign_in_page():
    st.subheader("Sign In")
    with st.form(key='sign_in_form'):
        username = st.text_input("Username", key="username_login")
        password = st.text_input("Password", type="password", key="password_login")
        submit_button = st.form_submit_button("Sign In")

    if submit_button:
        if username and password and validate_user(username, password):
            st.success("Successfully logged in!")
            st.session_state.logged_in = True
            st.session_state['username'] = username
            st.session_state.current_page = 'detection'
            st.experimental_rerun()
        else:
            st.warning("Invalid username or password.")


def sign_up_page():
    st.subheader("Sign Up")
    with st.form(key='sign_up_form'):
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        submit_button = st.form_submit_button("Sign Up")

    if submit_button:
        if new_username and new_password and save_user_to_db(new_username, new_password):
            st.success("Account created! Please sign in.")
            st.session_state.current_page = 'auth'
        else:
            st.warning("Username already exists or fields are invalid.")


if __name__ == "__main__":
    main()

#%%
