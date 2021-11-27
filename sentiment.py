import openai
import pandas as pd
from sklearn import metrics
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import flair
from sklearn.model_selection import train_test_split

#%% Functions to call OpenAI API with custom model parameters

def gpt_sentiment_class(call):
    openai.api_key = 'INSERT KEY HERE'
    
    response = openai.Completion.create(
      engine="davinci",
      prompt= call,
      temperature=0.3,
      max_tokens=60,
      top_p=1.0,
      frequency_penalty=0.5,
      presence_penalty=0.0,
      stop=["###"]
    )
    content = response.choices[0].text.split('.')
    print(content)
    return response.choices[0].text

def gpt_sentiment_class_tuned(call):
    openai.api_key = 'INSERT KEY HERE'
    
    response = openai.Completion.create(
      model= "curie:ft-user-wtzcwppgb4x7qt6el1c2u3lt-2021-11-08-14-19-51",
      prompt= call,
      temperature=0.3,
      max_tokens=1,
      top_p=1.0,
      frequency_penalty=0.5,
      presence_penalty=0.5,
      stop=["\n"]
    )
    content = response.choices[0].text.split('.')
    print(content)
    return response.choices[0].text

def func(x):
    if x > 0 :
        return 'Positive'
    else:
        return "Negative"


#%% https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format
# Load dataset, preprocess and perform train/test split

df = pd.read_csv("data/IMDB_Dataset.csv")

df["sentiment"] = [x.capitalize() for x in df["sentiment"]] 
X = list(df["review"])
y = list((df["sentiment"]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sent_tune_set = pd.DataFrame({"prompt": X_train, "completion": y_train})
pos = sent_tune_set[sent_tune_set["completion"] == "Positive"].sample(150)
neg = sent_tune_set[sent_tune_set["completion"] == "Negative"].sample(150)

#Number of samples to include in API call
n = 100

#%%

gpt_sent_pred = []
for review in X_test[0:n]:
    inp = "This is an IMDB movie review sentiment classifier\n\n\nReview: \"This is one of those movies that showcases a great actor's talent and also conveys a great story. It is one of Stewart's greatest movies. Barring a few historic errors it also does an excellent job of telling the story of the Spirit of St. Louis.\"\nSentiment: Positive\n###\nReview: \"This movie was a complete waste of time. The soundtrack was bad, story was lame and predictable, and the acting was terrible. One of the worst 25 movies I have ever seen. After the first ten minutes, the rest of the film was completely obvious.\"\nSentiment: Negative\n###\nReview: \"Great characters, great acting, great dialogue, incredible plot twists in plain language one of the best shows I've ever seen in my life. Do yourself a favor and watch this show, you won't regret it. This show re-writes the book on Sci-Fi!\"\nSentiment: Positive\n###\nReview: \"Ok,so.....guy gets bitten by a bat and then turns into a bat (well,sorta). I can only assume this made sense to SOMEONE at the time! Aren't bats supposed to fly, use radar, and eat bugs instead of attacking humans tho?\"\nSentiment: Negative\n###\nReview: \"" + review + "" + "\ \nSentiment:",
    gpt_sent_pred.append(gpt_sentiment_class(inp).strip())

#%%
gpt_sent_pred_tuned = []
for review in X_test[0:n]:
    inp = "This is an IMDB movie review sentiment classifier\n\n\nReview: \"This is one of those movies that showcases a great actor's talent and also conveys a great story. It is one of Stewart's greatest movies. Barring a few historic errors it also does an excellent job of telling the story of the Spirit of St. Louis.\"\nSentiment: Positive\n###\nReview: \"This movie was a complete waste of time. The soundtrack was bad, story was lame and predictable, and the acting was terrible. One of the worst 25 movies I have ever seen. After the first ten minutes, the rest of the film was completely obvious.\"\nSentiment: Negative\n###\nReview: \"Great characters, great acting, great dialogue, incredible plot twists in plain language one of the best shows I've ever seen in my life. Do yourself a favor and watch this show, you won't regret it. This show re-writes the book on Sci-Fi!\"\nSentiment: Positive\n###\nReview: \"Ok,so.....guy gets bitten by a bat and then turns into a bat (well,sorta). I can only assume this made sense to SOMEONE at the time! Aren't bats supposed to fly, use radar, and eat bugs instead of attacking humans tho?\"\nSentiment: Negative\n###\nReview: \"" + review + "" + "\ \nSentiment:",
    gpt_sent_pred_tuned.append(gpt_sentiment_class_tuned(inp).strip())
    
#%% Compute metrics and generate excel file

gpt_sent_acc = metrics.accuracy_score(y_test[0:n], list(gpt_sent_pred))
gpt_sent_rep = metrics.classification_report(y_test[0:n], list(gpt_sent_pred), output_dict=True)
gpt_sent_rep_df = pd.DataFrame(gpt_sent_rep).transpose().to_excel("GPT3_sent_rep.xlsx")

gpt_sent_tuned_acc = metrics.accuracy_score(y_test[0:n], list(gpt_sent_pred_tuned))
gpt_sent_tuned_rep = metrics.classification_report(y_test[0:n], list(gpt_sent_pred_tuned), output_dict=True)
gpt_sent_tuned_rep_df = pd.DataFrame(gpt_sent_tuned_rep).transpose().to_excel("GPT3_tuned_sent_rep.xlsx")

#%% Predict using Flair classifier

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

flair_pred_raw = []
for review in X_test[0:n]:
    s = flair.data.Sentence(review)
    flair_sentiment.predict(s)
    label = str(s.labels[0]).split()[0]
    flair_pred_raw.append(label)
    
flair_pred = [pred.lower().capitalize() for pred in flair_pred_raw]

#%% Compute Flair metrics

flair_sent_acc = metrics.accuracy_score(y_test[0:n], flair_pred)
flair_sent_rep = metrics.classification_report(y_test[0:n], flair_pred, output_dict=True)
flair_sent_rep_df = pd.DataFrame(flair_sent_rep).transpose().to_excel("Flair_sent_rep.xlsx")