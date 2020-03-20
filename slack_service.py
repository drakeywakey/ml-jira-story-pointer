import os
from dotenv import load_dotenv
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from jira_service import predict
from main import app

load_dotenv()
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