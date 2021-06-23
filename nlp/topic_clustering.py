import pandas as pd
from sklearn.cluster import KMeans
import nltk
import gensim

# nltk.download('punkt')
# nltk.download('stopwords')

# #### 1. importamos el dataset
df_sentiments = pd.read_csv("./nlp/sentiments.csv", delimiter=";")
df_sentiments = df_sentiments[["Sentiments"]]
print(df_sentiments)

# #### 2. Tokenizacion
df_sentiments["words_tk"] = df_sentiments.apply(lambda row: nltk.word_tokenize(row["Sentiments"]), axis=1)
df_sentiments["tokens"] = df_sentiments.apply(lambda row: len(row["words_tk"]), axis=1)
print(df_sentiments)

# #### 3. Remocion de Stop Words
stop_words = set(nltk.corpus.stopwords.words("english"))


def stopWords(words_list):
    filtered_sentence = []
    for w in words_list:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


df_sentiments["words_filtered"] = df_sentiments.apply(lambda row: stopWords(row["words_tk"]), axis=1)
df_sentiments["tokens_word_filtered"] = df_sentiments.apply(lambda row: len(row["words_filtered"]), axis=1)
print(df_sentiments)

# #### 4. Segundo stop removing
df_sentiments["words_tk_relevant"] = df_sentiments.apply(lambda row: [x for x in row["words_filtered"] if len(x) > 2],
                                                         axis=1)
df_sentiments["tokens_words_relevant"] = df_sentiments.apply(lambda row: len(row["words_tk_relevant"]), axis=1)
df_sentiments["words_tk_relevant_joined"] = df_sentiments.apply(lambda row: " ".join(row["words_tk_relevant"]), axis=1)
print(df_sentiments)

# #### 5. Word2Vec Process
model_w2v = gensim.models.Word2Vec(df_sentiments.words_tk_relevant.values.tolist(), min_count=0, vector_size=5, sg=1)


def avgVectorSentece(list_word, model):
    vector = [model_w2v.wv[x] for x in list_word]
    return sum(vector) / len(vector)


df_sentiments["result_w2v"] = df_sentiments.apply(lambda row: avgVectorSentece(row["words_tk_relevant"], model_w2v),
                                                  axis=1)
print(df_sentiments)

# #### 6. Clustering de vectores de oraciones
kmeans = KMeans(n_clusters=400, random_state=1).fit(df_sentiments.result_w2v.tolist())
df_sentiments["predicted_cluster"] = df_sentiments.apply(lambda row: kmeans.predict([row["result_w2v"].tolist()])[0],
                                                         axis=1)

print(df_sentiments.sort_values("predicted_cluster").head(100))
print(df_sentiments.shape)

df_sentiments[["Sentiments", "words_tk_relevant_joined", "predicted_cluster"]].to_csv(
    "./nlp/clustering_sentiments_w2v.csv", sep="\t", header=True, index=False)
