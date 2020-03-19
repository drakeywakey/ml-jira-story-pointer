from dotenv import load_dotenv
import os
import logging
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
import ssl as ssl_lib
import certifi
from service import predict

# Initialize a Flask app to host the events adapter
load_dotenv()
app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(os.getenv('SLACK_SIGNING_SECRET'), "/slack/events", app)

# Initialize a Web API client
slack_web_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))

@slack_events_adapter.on("app_mention")
def app_mention(payload):
    """
        When someone mentions the app in the 'what_the_hecky' channel, repeat
        back exactly what they say
    """
    event = payload.get("event", {})
    text = event.get("text")
    ticket = text.split(' ')[1]
    # todo: verify ticket is in format PXT-#

    slack_web_client.chat_postMessage(
        channel=os.environ.get("SLACK_CHANNEL", "general"),
        text=predict(ticket))

@app.route('/')
def welcome():
    return 'Running the yerba!'

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)