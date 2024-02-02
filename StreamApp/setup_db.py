
# setup_database.py
import sqlite3

conn = sqlite3.connect('drowsiness.db')
c = conn.cursor()
# Create predictions table
c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (date text, prediction text)''')
# Create users table
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username text PRIMARY KEY, password_hash text)''')
# Save (commit) the changes and close the connection
conn.commit()
conn.close()
# %%
