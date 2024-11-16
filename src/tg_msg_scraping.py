# Import necessary modules

from pathlib import Path               # For working with file paths
from dotenv import load_dotenv         # For loading environment variables from .env file
from datetime import datetime          # For working with dates and times

import os                              # For interacting with the operating system
import csv                             # For reading and writing CSV files

import fasttext                        # For language detection

from telethon import TelegramClient    # For interacting with the Telegram API
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import PeerChannel

# Load environment variables from .env file
dotenv_path = Path('config/.env')
load_dotenv(dotenv_path=dotenv_path)

# Extract Telegram API credentials from environment variables
api_id = int(os.getenv("TG_API_ID"))
api_hash = os.getenv("TG_API_HASH")

# Extract Telegram user information from environment variables
phone_number = os.getenv("TG_PHONE_NUMBER")
username = os.getenv("TG_USERNAME")

# Initialize a Telegram client with the provided API credentials
client = TelegramClient('Scraper', api_id, api_hash)

# Function to detect language of a message using FastText model
def detect_language(message):
    try:
        # Load FastText language detection model
        model = fasttext.load_model("models/lid.176.bin")
        # Predict the language of the message
        lang_detected = model.predict(message)
        return lang_detected[0][0].replace('__label__', '')  # Extract language label from prediction
    except Exception as e:
        print("An error occurred:", e)
        return None

# Function to get the group or channel entity from its ID
async def get_group_entity(client, id_):
    if id_.isdigit():   # Check if the ID is numeric
        entity = PeerChannel(int(id_))   # If so, it's a channel ID
    else:
        entity = id_   # Otherwise, it's assumed to be the entity itself
    return await client.get_entity(entity)

# Function to retrieve all replies to a specific post in a Telegram group/channel within a given date range
async def fetch_all_posts_replies_in_date_range(client, group, start_date, end_date):
    replies = []   # List to store retrieved replies
    async for post in client.iter_messages(group, reverse=True, offset_date=start_date):
        # Iterate over posts in reverse chronological order until the start date
        if str(post.date) > str(end_date):   # Check if the post date exceeds the end date
            break   # If so, stop iterating
        if isinstance(post.message, str) and post.replies is not None:
            # Check if the post has a message and has replies
            post.message = ' '.join(post.message.split('\n'))   # Normalize message formatting
            if post.replies.replies > 0:
                # If the post has replies, iterate over them
                async for reply in client.iter_messages(group, reply_to=post.id):
                    reply.message = ' '.join(reply.message.split('\n'))   # Normalize reply formatting
                    reply_lang = detect_language(reply.message)   # Detect language of the reply
                    reply_dict = reply.to_dict()   # Convert reply to dictionary
                    # Add additional fields to the reply dictionary
                    reply_dict['message_lang'] = reply_lang
                    reply_dict['post_id'] = post.id
                    reply_dict['post_message'] = post.message
                    reply_dict['post_date'] = post.date
                    replies.append(reply_dict)   # Append the reply dictionary to the list
    return replies   # Return the list of replies

# Function to write messages to a CSV file
def messages_to_csv(messages, filename, group_id, group_username):
    keys = ['group_id', 'group_username', 'post_id', 'post_date', 'post_msg', 'msg_id', 'msg_date', 'msg_lang', 'msg']
    file_exists = os.path.exists(filename)   # Check if the CSV file already exists
    # If the file exists, check if its headers match the expected keys
    if file_exists:
        with open(filename, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            file_keys = next(reader, None)   # Read the headers from the file
        # If the headers do not match, print a message and return
        if file_keys != keys:
            print("Headers in the existing file do not match the current headers.")
            return
    # Open the CSV file for writing
    with open(filename, 'a', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=keys)
        # If the file doesn't exist, write the headers
        if not file_exists:
            writer.writeheader()
        # Iterate over messages and write them to the CSV file
        for message in messages:
            if message["_"] == "Message":   # Check if the item is a message
                # Write message details to the CSV file
                writer.writerow({
                    'group_id': group_id,
                    'group_username': "@" + group_username,
                    'post_id': message["post_id"],
                    'post_date': message["post_date"],
                    'post_msg': message["post_message"],
                    'msg_id': message["id"],
                    'msg_date': message["date"],
                    'msg_lang': message["message_lang"],
                    'msg': message["message"]
                })

# Async function to orchestrate the scraping process
async def main():
    await client.start()   # Start the Telegram client
    print("Client Created")
    if not await client.is_user_authorized():   # Check if the user is authorized
        # If not, request authorization code and sign in
        await client.send_code_request(phone_number)
        try:
            await client.sign_in(phone_number, input('Enter the code: '))
        except SessionPasswordNeededError:
            await client.sign_in(password=input('Password: '))
    # List of groups or channels to scrape messages from
    group_list = ["@truexanewsua", "@UaOnlii", "@okoo_ukr", "@uniannet", "@ssternenko",
                  "@DeepStateUA", "@bozhejakekonchene", "@operativnoZSU", "@kyivoperat", "@karas_evgen"]
    # Iterate over each group or channel
    for group in group_list:
        group = await get_group_entity(client, group)   # Get group entity
        # Define start and end dates for the scraping period
        start_date = datetime(2024, 2, 1, 0, 0, 0)
        end_date = datetime(2024, 2, 2, 0, 0, 0)
        # Fetch messages and replies within the specified date range
        messages = await fetch_all_posts_replies_in_date_range(client, group, start_date, end_date)
        # Write messages to CSV file
        messages_to_csv(messages, "src/Datasets/raw/02_February/TelegramChannelMessages_01022024.csv", group.id, group.username)

# Run the main async function
with client:
    client.loop.run_until_complete(main())
