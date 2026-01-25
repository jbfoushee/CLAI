import os
from flask import Flask, request, g
import sqlite3


app = Flask(__name__)
app.config['DATABASE'] = 'users.db'


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
    return g.db

# Ensure DB connection is closed after request
@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()



@app.route('/user')
def get_user_profile():
    user_id = request.args.get('id')
    if user_id is None:
        return {"error": "Missing 'id' parameter."}, 400
    try:
        user_id_int = int(user_id)
    except ValueError:
        return {"error": "Invalid 'id' parameter. Must be an integer."}, 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id_int,))
        user_row = cursor.fetchone()
        if user_row is None:
            return {"error": "User not found."}, 404
        user_profile = process_data(user_row)
        response = format_response(user_profile)
        return response
    except sqlite3.Error as e:
        return {"error": f"Database error: {e}"}, 500


# (Assume process_data and format_response exist elsewhere)
