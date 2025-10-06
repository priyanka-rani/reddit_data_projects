import praw
import pandas as pd
import re
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os
from dotenv import load_dotenv

# ------------------------------
# Reddit API Authentication
# ------------------------------

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text):
    text = emoji.replace_emoji(text, replace='')  # remove emojis
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove mentions
    text = re.sub(r"#\w+", "", text)              # remove hashtags
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip()      # collapse spaces
    return text

# ------------------------------
# Load Toxicity Model
# ------------------------------
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ------------------------------
# Toxicity Prediction Function
# ------------------------------
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
    return dict(zip(labels, scores))

# ------------------------------
# Collect Reddit Comments
# ------------------------------
def collect_reddit_comments(subreddit_name="AskReddit", limit=50):
    comments = []
    subreddit = reddit.subreddit(subreddit_name)
    for comment in tqdm(subreddit.comments(limit=limit), desc=f"Collecting from r/{subreddit_name}"):
        text = clean_text(comment.body)
        if len(text) > 5:  # skip empty/short comments
            toxicity = predict_toxicity(text)
            comments.append({
                "subreddit": subreddit_name,
                "comment_id": comment.id,
                "author": str(comment.author),
                "clean_text": text,
                **toxicity
            })
    return comments

# ------------------------------
# Main Function
# ------------------------------
if __name__ == "__main__":
    subreddit_list = ["AskReddit", "worldnews", "funny"]
    all_comments = []

    for sub in subreddit_list:
        data = collect_reddit_comments(sub, limit=100)
        all_comments.extend(data)

    # Save to CSV
    df = pd.DataFrame(all_comments)
    df.to_csv("data/reddit_toxicity_dataset.csv", index=False)
    print("\n✅ Saved reddit_toxicity_dataset.csv")