import os
from flask import Flask
import ssl as ssl_lib
import certifi

# Initialize a Flask app to host the events adapter
app = Flask(__name__)

import slack_service

@app.route('/')
def welcome():
    return 'Running the yerba!'

if __name__ == "__main__":
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)