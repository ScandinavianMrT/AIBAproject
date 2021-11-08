import os
import openai
import pandas as pd
import numpy as np
import requests
from nltk.translate.bleu_score import sentence_bleu
import string
import re
from rouge import Rouge
from sklearn.model_selection import train_test_split
import re

#%% https://www.kaggle.com/pariza/bbc-news-summary

classes = os.listdir("C:/Users/T-bone/Desktop/AIBA/data/bbc-news-summary/BBC News Summary/News Articles/")
Articles_dir = 'C:/Users/T-bone/Desktop/AIBA/data/bbc-news-summary/BBC News Summary/News Articles/'
Summaries_dir = 'C:/Users/T-bone/Desktop/AIBA/data/bbc-news-summary/BBC News Summary/Summaries/'

#%%
articles = []
summaries = []
file_arr = []
for cls in classes:
    files = os.listdir(Articles_dir + cls + "/")
    for file in files:
        article_file_path = Articles_dir + cls + '/' + file
        summary_file_path = Summaries_dir + cls + '/' + file
        try:
            with open (article_file_path,'r') as f:
                articles.append('.'.join([line.rstrip() for line in f.readlines()]))
            with open (summary_file_path,'r') as f:
                summaries.append('.'.join([line.rstrip() for line in f.readlines()]))
            file_arr.append(cls + '/' + file)
        except:
            pass
            
dataset = pd.DataFrame({'File_path':file_arr,'Articles': articles,'Summaries':summaries})
mask = (dataset['Articles'].str.len() < 1350) & (dataset['Articles'].str.len() > 100)
df = dataset.loc[mask].sample(400)
X_train, X_test, y_train, y_test = train_test_split(df["Articles"], df["Summaries"], test_size=0.25, random_state=42)
tune_set_sum = pd.DataFrame({"Articles": X_train, "Summaries": y_train}).to_excel("fine_tune_sum.xlsx")

n = 2

#%%
def gpt_summary(text):
    prompt="Savvy searchers fail to spot ads.. Internet search engine users are an odd mix of naive and sophisticated, suggests a report into search habits... The report by the US Pew Research Center reveals that 87% of searchers usually find what they were looking for when using a search engine. It also shows that few can spot the difference between paid-for results and organic ones. The report reveals that 84% of net users say they regularly use Google, Ask Jeeves, MSN and Yahoo when online... Almost 50% of those questioned said they would trust search engines much less, if they knew information about who paid for results was being hidden. According to figures gathered by the Pew researchers the average users spends about 43 minutes per month carrying out 34 separate searches and looks at 1.9 webpages for each hunt. A significant chunk of net users, 36%, carry out a search at least weekly and 29% of those asked only look every few weeks. For 44% of those questioned, the information they are looking for is critical to what they are doing and is information they simply have to find... Search engine users also tend to be very loyal and once they have found a site they feel they can trust tend to stick with it. According to Pew Research 44% of searchers use just a single search engine, 48% use two or three and a small number, 7%, consult more than three sites. Tony Macklin, spokesman for Ask Jeeves, said the results reflected its own research which showed that people use different search engines because the way the sites gather information means they can provide different results for the same query. Despite this liking for search sites half of those questioned said they could get the same information via other routes. A small number, 17%, said they wouldn't really miss search engines if they did not exist. The remaining 33% said they could not live without search sites. More than two-thirds of those questioned, 68%, said they thought that the results they were presented with were a fair and unbiased selection of the information on a topic that can be found on the net. Alongside the growing sophistication of net users is a lack of awareness about paid-for results that many search engines provide alongside lists of websites found by indexing the web. Of those asked, 62% were unaware that someone has paid for some of the results they see when they carry out a search. Only 18% of all searchers say they can tell which results are paid for and which are not. Said the Pew report: \"This finding is ironic, since nearly half of all users say they would stop using search engines if they thought engines were not being clear about how they presented paid results.\" Commenting Mr Macklin said sponsored results must be clearly marked and though they might help with some queries user testing showed that people need to be able to spot the difference.\n###\n\ntl;dr:\n\nAlmost 50% of those questioned said they would trust search engines much less, if they knew information about who paid for results was being hidden. Said the Pew report: \"This finding is ironic, since nearly half of all users say they would stop using search engines if they thought engines were not being clear about how they presented paid results. \"Tony Macklin, spokesman for Ask Jeeves, said the results reflected its own research which showed that people use different search engines because the way the sites gather information means they can provide different results for the same query. Internet search engine users are an odd mix of naive and sophisticated, suggests a report into search habits. The report by the US Pew Research Center reveals that 87% of searchers usually find what they were looking for when using a search engine. A small number, 17%, said they wouldn't really miss search engines if they did not exist. Alongside the growing sophistication of net users is a lack of awareness about paid-for results that many search engines provide alongside lists of websites found by indexing the web. Despite this liking for search sites half of those questioned said they could get the same information via other routes.\n###\n\nEverton's Weir cools Euro hopes..Everton defender David Weir has played down talk of European football, despite his team lying in second place in the Premiership after beating Liverpool...Weir told BBC Radio Five Live: \"We don't want to rest on our laurels and say we have achieved anything yet. \"I think you start taking your eye off the ball if you make statements and look too far into the future. \"If you start making predictions you soon fall back into trouble. The only thing that matters is the next game.\" He said: \"We are looking after each other and hard work goes a long way in this league. We have definitely shown that. \"Also injuries and suspensions haven't cost us too badly and we have a lot of self-belief around the place.\"\n###\n\ntl;dr:\nEverton defender David Weir has played down talk of European football, despite his team lying in second place in the Premiership after beating Liverpool.\"I think you start taking your eye off the ball if you make statements and look too far into the future.\"If you start making predictions you soon fall back into trouble.\"Also injuries and suspensions haven't cost us too badly and we have a lot of self-belief around the place.\"\n###\n\n" + text + "\n###" + "\n\n" + "tl;dr:",
    print(prompt)
    openai.api_key = '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'
    engine_list = openai.Engine.list() 
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=0.3,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"]
        )
    return response.choices[0].text

def gpt_summary_tuned(text):
    prompt="Savvy searchers fail to spot ads.. Internet search engine users are an odd mix of naive and sophisticated, suggests a report into search habits... The report by the US Pew Research Center reveals that 87% of searchers usually find what they were looking for when using a search engine. It also shows that few can spot the difference between paid-for results and organic ones. The report reveals that 84% of net users say they regularly use Google, Ask Jeeves, MSN and Yahoo when online... Almost 50% of those questioned said they would trust search engines much less, if they knew information about who paid for results was being hidden. According to figures gathered by the Pew researchers the average users spends about 43 minutes per month carrying out 34 separate searches and looks at 1.9 webpages for each hunt. A significant chunk of net users, 36%, carry out a search at least weekly and 29% of those asked only look every few weeks. For 44% of those questioned, the information they are looking for is critical to what they are doing and is information they simply have to find... Search engine users also tend to be very loyal and once they have found a site they feel they can trust tend to stick with it. According to Pew Research 44% of searchers use just a single search engine, 48% use two or three and a small number, 7%, consult more than three sites. Tony Macklin, spokesman for Ask Jeeves, said the results reflected its own research which showed that people use different search engines because the way the sites gather information means they can provide different results for the same query. Despite this liking for search sites half of those questioned said they could get the same information via other routes. A small number, 17%, said they wouldn't really miss search engines if they did not exist. The remaining 33% said they could not live without search sites. More than two-thirds of those questioned, 68%, said they thought that the results they were presented with were a fair and unbiased selection of the information on a topic that can be found on the net. Alongside the growing sophistication of net users is a lack of awareness about paid-for results that many search engines provide alongside lists of websites found by indexing the web. Of those asked, 62% were unaware that someone has paid for some of the results they see when they carry out a search. Only 18% of all searchers say they can tell which results are paid for and which are not. Said the Pew report: \"This finding is ironic, since nearly half of all users say they would stop using search engines if they thought engines were not being clear about how they presented paid results.\" Commenting Mr Macklin said sponsored results must be clearly marked and though they might help with some queries user testing showed that people need to be able to spot the difference.\n###\n\ntl;dr:\n\nAlmost 50% of those questioned said they would trust search engines much less, if they knew information about who paid for results was being hidden. Said the Pew report: \"This finding is ironic, since nearly half of all users say they would stop using search engines if they thought engines were not being clear about how they presented paid results. \"Tony Macklin, spokesman for Ask Jeeves, said the results reflected its own research which showed that people use different search engines because the way the sites gather information means they can provide different results for the same query. Internet search engine users are an odd mix of naive and sophisticated, suggests a report into search habits. The report by the US Pew Research Center reveals that 87% of searchers usually find what they were looking for when using a search engine. A small number, 17%, said they wouldn't really miss search engines if they did not exist. Alongside the growing sophistication of net users is a lack of awareness about paid-for results that many search engines provide alongside lists of websites found by indexing the web. Despite this liking for search sites half of those questioned said they could get the same information via other routes.\n###\n\nEverton's Weir cools Euro hopes..Everton defender David Weir has played down talk of European football, despite his team lying in second place in the Premiership after beating Liverpool...Weir told BBC Radio Five Live: \"We don't want to rest on our laurels and say we have achieved anything yet. \"I think you start taking your eye off the ball if you make statements and look too far into the future. \"If you start making predictions you soon fall back into trouble. The only thing that matters is the next game.\" He said: \"We are looking after each other and hard work goes a long way in this league. We have definitely shown that. \"Also injuries and suspensions haven't cost us too badly and we have a lot of self-belief around the place.\"\n###\n\ntl;dr:\nEverton defender David Weir has played down talk of European football, despite his team lying in second place in the Premiership after beating Liverpool.\"I think you start taking your eye off the ball if you make statements and look too far into the future.\"If you start making predictions you soon fall back into trouble.\"Also injuries and suspensions haven't cost us too badly and we have a lot of self-belief around the place.\"\n###\n\n" + text + "\n###" + "\n\n" + "tl;dr:",
    print(prompt)
    openai.api_key = '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'
    engine_list = openai.Engine.list() 
    response = openai.Completion.create(
        model="curie:ft-user-wtzcwppgb4x7qt6el1c2u3lt-2021-11-05-14-11-49",
        prompt= prompt,
        temperature=0.3,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"]
        )
    return response.choices[0].text

def deep_summary(text):
    r = requests.post(
        "https://api.deepai.org/api/summarization",
        data={
            'text': text,
        },
        headers={'api-key': 'd7f4cc16-dc53-4f3c-8a5f-b8cb9ece618f'}
    )
    return r.json().get("output")

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

def corr(s):
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', s))

#%%

gpt_sum_pred = []
for text in X_test[0:n]:
    gpt_sum_pred.append(gpt_summary(text).strip())
    
#%%

gpt_sum_pred_tuned = []
for text in X_test[0:n]:
    gpt_sum_pred_tuned.append(gpt_summary_tuned(text).strip())

gpt_sum_pred_tuned = [corr(text) for text in gpt_sum_pred_tuned]

#%%

deep_responses = []
for text in X_test[0:n]:
    deep_responses.append(deep_summary(text))
    
#%% Get BLEU-scores
 
refs = [sentence.split() for sentence in remove_punc(y_test[0:n])]
gpt_candidates = [sentence.split() for sentence in gpt_sum_pred]
gpt_candidates_tuned = [sentence.split() for sentence in gpt_sum_pred_tuned]
deep_candidates = [sentence.split() for sentence in deep_responses]

gpt_bleu = BLEU(refs, gpt_candidates)
gpt_bleu_tuned = BLEU(refs, gpt_candidates_tuned)
deep_bleu = BLEU(refs, deep_candidates)

#%% https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460

# Calculate Rouge precision based on bigrams and get f1 score
gpt_rouge = Rouge().get_scores(gpt_sum_pred, list(y_test[0:n]), avg=True).get("rouge-2").get("f")
gpt_rouge_tuned = Rouge().get_scores(gpt_sum_pred_tuned, list(y_test[0:n]), avg=True).get("rouge-2").get("f")
deep_rouge = Rouge().get_scores(deep_responses, list(y_test[0:n]), avg=True).get("rouge-2").get("f")

