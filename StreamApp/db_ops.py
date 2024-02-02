import sqlite3
from datetime import datetime

def save_prediction_to_db(prediction):
    conn = sqlite3.connect("drowsiness.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO predictions (date, prediction) VALUES (?, ?)", (timestamp, prediction))
    conn.commit()
    conn.close()

def validate_user(username, password):
    with sqlite3.connect('drowsiness.db') as conn:
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        if user and user[0] == password:
            return True
    return False


def save_user_to_db(username, password):
    try:
        with sqlite3.connect('drowsiness.db') as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                      (username, password))
            conn.commit()
    except sqlite3.IntegrityError:
        return False  # Username already exists
    return True

def history():
    conn = sqlite3.connect('drowsiness.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY date DESC")
    predictions = c.fetchall()
    conn.close()
    return predictions