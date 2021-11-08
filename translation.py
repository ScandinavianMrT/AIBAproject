import openai
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import string
import re
import numpy as np
from googletrans import Translator
from rouge import Rouge
from sklearn.model_selection import train_test_split
import json

#%% Define functions

def translate(call):
    openai.api_key = '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'
    
    response = openai.Completion.create(
      engine="davinci",
      prompt = call,
      temperature=0.5,
      
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
      )
    content = response.choices[0].text.split('.')
    print(content)
    return response.choices[0].text

def translate_tuned(call):
    openai.api_key = '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'
    
    response = openai.Completion.create(
      model="curie:ft-user-wtzcwppgb4x7qt6el1c2u3lt-2021-11-05-13-56-04",
      prompt = call,
      temperature=0.5,
      
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
      )
    content = response.choices[0].text.split('.')
    print(content)
    return response.choices[0].text

def remove_punc(corpus):
    texts = []
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    for text in corpus:
        texts.append(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key)))
    return texts

def BLEU(refs, candidates):
    scores = []
    for cand in candidates:
        score = sentence_bleu(refs, cand, weights=(0.25, 0.25, 0.25, 0.25)) 
        scores.append(score)

    return np.mean(scores)     

#%% import dataset and preprocess
#https://www.kaggle.com/harishreddy18/english-to-french

df_eng = pd.read_csv('data/small_vocab_en.csv', sep='delimiter', header = None)
sentences_eng = remove_punc(list(df_eng[0:400][0]))
df_fr = pd.read_csv('data/small_vocab_fr.csv', sep='delimiter', header = None)
sentences_fr = remove_punc(list(df_fr[0:400][0]))
df = pd.DataFrame({"English":sentences_eng,"French": sentences_fr})
X_train, X_test, y_train, y_test = train_test_split(df["English"], df["French"], test_size=0.25, random_state=42)
tune_set_trans = pd.DataFrame({"English": X_train, "French": y_train}).to_excel("fine_tune_trans.xlsx")

n = 100

#%% Get GPT-3 predictions

gpt_trans_pred = []

for sentence in X_test[0:n]:
    inp = "English: I do not speak French.\nFrench: Je ne parle pas français.\n\nEnglish: See you later!\nFrench: À tout à l'heure!\n\nEnglish: Where is a good restaurant?\nFrench: Où est un bon restaurant?\n\nEnglish: What rooms do you have available?\nFrench: Quelles chambres avez-vous de disponible?\n\nEnglish: " + sentence + "\nFrench:"
    gpt_trans_pred.append(translate(inp).strip())
    
#%%
gpt_trans_pred_tuned = []

for sentence in X_test[0:n]:
    inp = "English: I do not speak French.\nFrench: Je ne parle pas français.\n\nEnglish: See you later!\nFrench: À tout à l'heure!\n\nEnglish: Where is a good restaurant?\nFrench: Où est un bon restaurant?\n\nEnglish: What rooms do you have available?\nFrench: Quelles chambres avez-vous de disponible?\n\nEnglish: " + sentence + "\nFrench:"
    gpt_trans_pred_tuned.append(translate_tuned(inp).strip())
    
#%% Get Google Translate predictions
ggl_pred = []
translator = Translator()

for sentence in X_test[0:n]:
    pred = translator.translate(sentence, dest = "fr").text
    ggl_pred.append(pred)

#%% Get BLEU-scores for both apps

refs = [sentence.split() for sentence in remove_punc(y_test[0:n])]
gpt_candidates = [sentence.split() for sentence in gpt_trans_pred]
gpt_candidates_tuned = [sentence.split() for sentence in gpt_trans_pred_tuned]
ggl_candidates = [sentence.split() for sentence in ggl_pred]

gpt_bleu_trans = BLEU(refs, gpt_candidates)
gpt_bleu_trans_tuned = BLEU(refs, gpt_candidates_tuned)
ggl_bleu_trans = BLEU(refs, ggl_candidates)

#%%
gpt_rouge_trans = Rouge().get_scores(gpt_trans_pred, list(y_test[0:n]), avg=True).get("rouge-2").get("f")
gpt_rouge_trans_tuned = Rouge().get_scores(gpt_trans_pred_tuned, list(y_test[0:n]), avg=True).get("rouge-2").get("f")
ggl_rouge_trans = Rouge().get_scores(ggl_pred, list(y_test[0:n]), avg=True).get("rouge-2").get("f")
