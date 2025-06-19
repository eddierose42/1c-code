import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from gensim.models import Word2Vec
from gensim.matutils import unitvec
import spacy
from nltk.stem import WordNetLemmatizer
import networkx as net
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
pd.set_option('mode.copy_on_write', True)
rng = np.random.default_rng(seed=42)

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

class text_processing:
    def load_corpus(name):
        df = pd.read_csv("data/corpus/{}_corpus.txt".format(name))
        df.set_index("Unnamed: 0", drop=True, inplace=True)
        df.index.rename(None, inplace=True)
        series = df["0"].apply(eval)
        return series

    def apply_to_subset(data, function, return_series=False, **kwargs):
        series = pd.Series(data)
        processed = series.apply(function, **kwargs)
        if return_series:
            return processed
        else:
            return processed.to_list()

    def lemmatize(doc, remove_stopwords=True, remove_punctuation=True, return_string=False): # takes in a nlp(text) object
        lemmatized_tokens = [token.lemma_ for token in doc if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha)]
        if return_string:
            lemmatized_text = ' '.join(lemmatized_tokens)
            return lemmatized_text
        else:
            return lemmatized_tokens

    def preprocess_corpus(corpus, keep_paragraphs=False, remove_stopwords=True, remove_punctuation=True, to_return="lemmatized", return_series=True): 
        if keep_paragraphs:
            lower = corpus.apply(text_processing.apply_to_subset, function=str.lower)
            tokenised = lower.apply(text_processing.apply_to_subset, function=nlp)
            lemmatized = tokenised.apply(text_processing.apply_to_subset, function=text_processing.lemmatize, remove_stopwords=remove_stopwords, remove_punctuation=remove_punctuation)
        else:
            joined = corpus.apply(" ".join)
            lower = joined.apply(str.lower)
            tokenised = lower.apply(nlp)
            lemmatized = tokenised.apply(text_processing.lemmatize, remove_stopwords=remove_stopwords, remove_punctuation=remove_punctuation)
        
        if to_return == "lemmatized":
            return_obj = lemmatized
        elif to_return == "tokenised":
            return_obj = tokenised

        if not return_series:
            return_obj = return_obj.to_list()
        return return_obj

class VAD_analysis:
    def __init__(self, polar=False):
        self.df = pd.read_csv("data/lexicon/NRC-VAD-Lexicon-v2.1.txt", sep="\t")
        self.df.set_index("term", inplace=True, drop=True)
        self.analyse = lambda word, var: self.loc[word, var] if word in self.index else None
        if polar:
            self.subset_polar()

    def __getitem__(self, key):
        return self.df[key]
    
    def __getattr__(self, attr):
        return getattr(self.df, attr)
    
    def analyse_df(self, df, target, return_df=True, return_results=False):
        if not list(self.columns) in list(df.columns):
            for var in self.columns:
                df[var] = df[target].apply(self.analyse, var=var)

        if return_results:
            results = pd.DataFrame()
            results.index = df.index.value_counts().index

            for var in self.columns:
                row = []
                for idx in results.index:
                    val = np.nanmean(df.loc[idx][var])
                    row.append(val)
                results[var] = row

        if return_df and return_results:
            return df, results
        elif return_results:
            return results
        else:
            return df

    def subset_polar(self, size=0.33):
        def find_polar(num):
            if abs(num) > size:
                return num

        for var in self.columns:
            self.df[var] = self[var].apply(find_polar)
        self.dropna(inplace=True)

class word_embedding:
    model = None

    def get_word2vec_model(source, merge=False, output=False):
        def load_model(name, merge=False, output=output):
            if merge:
                path = "data/word2vec/{}.model".format("_".join(name))
            else:
                path = "data/word2vec/{}.model".format(name)

            if output: print(path, os.path.exists(path))
            
            if os.path.exists(path):
                model = Word2Vec.load(path)
                return model
            else:
                if merge:
                    corpora = []             
                    for i in name:
                        raw = text_processing.load_corpus(i)
                        corpora.append(text_processing.preprocess_corpus(raw, return_series=True))
                    processed = pd.Series()
                    for i in corpora:
                        processed = pd.concat([processed, i])
                else:
                    raw = text_processing.load_corpus(name)
                    processed = text_processing.preprocess_corpus(raw, return_series=True)
                model = Word2Vec(sentences=processed.to_list())
                model.save(path)
                return model
        
        if type(source) is list and not merge:
            models = []
            for i in source:
                models.append(load_model)
            return models
        else:
            model = load_model(source, merge)
            if word_embedding.model is None:
                word_embedding.model = model
            return model

    def get_nearest_neighbours(target_groups, k=100, model=None):
        if model is None:
            model = word_embedding.model

        results = pd.DataFrame()
        for target in target_groups:
            if target in model.wv.key_to_index:
                data = model.wv.most_similar_cosmul(target, topn=k)
                results[target] = data
        results = results.melt()

        def get_col(val, idx):
            return val[idx]

        df = pd.DataFrame()
        df["word"] = results["value"].apply(get_col, idx=0)
        df["word_lemmatized"] = df["word"].apply(lemmatizer.lemmatize) ## only changes 6 [on test corpus]
        df["dist"] = results["value"].apply(get_col, idx=1)
        df.index = results["variable"] ## CHANGE TO MULTIINDEX??
        df.index.rename(None, inplace=True)
        return df

    def get_word_vector(word_list, model=None, normalize=True, ignore_lemmas=[], output=False, return_df=False, return_count=False):
        if model is None:
            model = word_embedding.model

        df = pd.DataFrame()
        if not type(word_list) is pd.Series:
            word_list = pd.Series(word_list)
        df["word"] = word_list

        def get_count(word):
            if word in model.wv:
                return model.wv.get_vecattr(word, "count")

        def get_vec(word):
            if word in model.wv:
                return model.wv[word]

        df["count"] = df["word"].apply(get_count)
        df["vector"] = df["word"].apply(get_vec)
        lemmas_found = []
        words_found = []

        for word in model.wv.index_to_key:
            for lemma in df[df["count"].isna()]["word"].values:
                if lemma in word and not lemma in ignore_lemmas:
                    if word.find(lemma) == 0:
                        if output:  print(lemma, ">", word)
                        if lemma in lemmas_found:
                            words_found[lemmas_found.index(lemma)].append(word)
                        else:
                            lemmas_found.append(lemma)
                            words_found.append([word])

        for i in range(len(lemmas_found)):
            total = 0
            for word in words_found[i]:
                var = model.wv.get_vecattr(word, "count")
                total += var
            count = total / len(words_found[i])

            total = 0
            for word in words_found[i]:
                total += model.wv[word]
            vector = total / len(words_found[i])

            df.loc[df["word"] == lemmas_found[i], "count"] = count
            df.at[df.index[df["word"] == lemmas_found[i]][0], "vector"] = vector

        final_vector = (df["vector"] * df["count"]).sum() / df["count"].sum()
        
        if normalize:
            final_vector = unitvec(final_vector)
        
        if return_df:
            return final_vector, df
        elif return_count:
            return final_vector, df["count"].sum()
        else:
            return final_vector

    def visualise_vectors_scatter(words, model=model, return_2d=False, offset=0.02):
        flatten = TSNE(n_components=2, random_state=0, perplexity=15, n_iter=5000)
        neighbour_vecs_2d = flatten.fit_transform(model.wv[words])
        plt.figure(figsize=(14,10))

        for i, word in enumerate(words):
            x, y = neighbour_vecs_2d[i,:]
            plt.scatter(x,y)
            plt.text(x+offset, y+offset, word, fontsize=11)

        plt.title("Visualisation of word embeddings")
        plt.show()
        if return_2d: return neighbour_vecs_2d

    def visualise_vectors_network(words, model=model, k=5, node_col="gold", return_graph=False):
        graph = net.Graph()
        for word in words:
            graph.add_node(word)

        for word in words:
            similar =  model.wv.most_similar(word, topn=k)
            for neighbour, similarity in similar:
                if neighbour in words:
                    graph.add_edge(word, neighbour, weight=similarity)

        plt.figure(figsize=(12,10))
        positions = net.spring_layout(graph, k=0.5, iterations=50)
        edges = graph.edges(data=True)
        weights = [d['weight'] * 5 for (var1, var2, d) in edges]
        net.draw_networkx_nodes(graph, positions, node_color=node_col, edgecolors="black", linewidths=1, node_size=2000)
        net.draw_networkx_edges(graph, positions, width=weights, alpha=0.6)
        net.draw_networkx_labels(graph, positions, font_size=10)
        plt.show()
        if return_graph: return graph

class paragraph_level_vad:
    def analyse(corpus, wordlist, VAD_obj=None):
        if VAD_obj is None:
            VAD_obj = VAD
        corpus = corpus.apply(text_processing.apply_to_subset, function=" ".join)
        all_paragraphs = pd.Series(np.concatenate(corpus.to_list()))
        relevant_paragraphs = all_paragraphs.apply(utilities.get_relevant, wordlist=wordlist).dropna().drop_duplicates()
        relevant_paragraphs_split = relevant_paragraphs.apply(str.split, " ")

        results = pd.DataFrame()
        results["text"] = relevant_paragraphs
        results["text_split"] = relevant_paragraphs_split
        results["len"] = relevant_paragraphs_split.apply(len)
        clean_none = lambda dat: np.nan if dat is None else dat

        for var in VAD_obj.columns:
            data = relevant_paragraphs_split.apply(text_processing.apply_to_subset, function=VAD_obj.analyse, var=var)
            data = data.apply(text_processing.apply_to_subset, function=clean_none)
            results[var] = data.apply(np.nanmean).dropna()

        drop_idx = results.index[np.unique(np.where(pd.isnull(results))[0])]
        results.drop(drop_idx, inplace=True)
        return results

    def get_article_index(df, corpus, target_col="text_split"):
        def instance(row):
            text = row[target_col]
            data = article_idx_from_search.loc[row.name]
            count = 0
            flag = 0

            if len(data) > 1:
                while text in seen_before[count]:        
                    if count == len(seen_before)-1:
                        flag = 1
                        break
                    else:
                        count += 1
                if flag:
                    seen_before.append([text])
                    count += 1
                else:
                    seen_before[count].append(text)     
                if count >= len(data): #366 occurrences in test corpus. maybe same phrase multiple times in same article. what to do?
                    return None
            
            return data[count]

        seen_before = [[]]
        temp_df = pd.DataFrame()
        temp_df["corpus"] = corpus
        search = lambda term: temp_df.apply(utilities.find, axis=1, col="corpus", searchterm=term).dropna().values
        article_idx_from_search = df[target_col].apply(search)
        article_idx_final = df.apply(instance, axis=1)
        return article_idx_final

    def compare(dfs, func=np.mean, names=[None, None], return_df=True, return_comparison=True):
        compare_df = pd.DataFrame()
        compare_df.index = VAD.columns

        for i in range(len(dfs)):
            if names[i] is None:
                name = "group{}".format(i+1)
            else:
                name = names[i]
            compare_df[name] = None
            for var in compare_df.index:
                compare_df.loc[var, name] = func(dfs[i][var])
        
        if return_df and not return_comparison:
            return compare_df
        
        elif return_df and return_comparison and len(dfs) == 2:
            return compare_df, compare_df[compare_df.columns[0]] - compare_df[compare_df.columns[1]]

        elif return_comparison and len(dfs) == 2:
            return compare_df[compare_df.columns[0]] - compare_df[compare_df.columns[1]] 

class data_analysis:
    def permutation_test_difference(group1, group2, test_types, function, n=1000, output=True, significance_level=0.05, test_significance=True, return_proportions=False):
        n_group1 = len(group1)

        pooled = np.concatenate([group1, group2])
        shuffled = rng.permutation(pooled)

        permutation_results = pd.DataFrame()
        for var in test_types.keys():
            permutation_results[var] = np.zeros(n)
        test_results = []
        proportions = []

        for i in range(n):
            shuffled = rng.permutation(pooled)
            group1_permutation = shuffled[:n_group1]
            group2_permutation = shuffled[n_group1:]
            group1_results = function(group1_permutation)
            group2_results = function(group2_permutation)
            
            mean_difference_series = group1_results[test_types.keys()].mean() - group2_results[test_types.keys()].mean()
            permutation_results.loc[i, mean_difference_series.index] = mean_difference_series.values

        if output and test_significance: print("Conducting permutation test:")

        for var in test_types.keys():
            actual_difference = function(group2)[var].mean() - function(group1)[var].mean()
            if test_types[var] == "greater":
                n_as_high = np.count_nonzero(permutation_results[var] <= actual_difference)
            elif test_types[var] == "less":
                n_as_high = np.count_nonzero(permutation_results[var] >= actual_difference)
            proportion = n_as_high / n
            proportions.append(proportion)

            if test_significance:
                if proportion < significance_level:
                    if output: print("\tThere is enough evidence to reject the null hypothesis for", var, "(p: {})\n".format(proportion))
                    test_results.append(1)
                else:
                    if output: print("\tThere is insufficient evidence to reject the null hypothesis for", var, "(p: {})\n".format(proportion))
                    test_results.append(0)
        
        if test_significance and return_proportions:
            return test_results, proportions
        
        elif test_significance:
            return test_results
        
        elif return_proportions:
            return proportions

class utilities:
    def get_relevant(text, wordlist):
        for i in wordlist:
            if i in text:
                return text

    def find(row, col, searchterm):
        if searchterm in row[col]:
            return row.name

    def adjust_xlabel(num=4, plt=plt):
        values, labels = plt.xticks()
        new = [[], []]
        for i in range(0, len(values), num):
            new[0].append(values[i])
            new[1].append(labels[i])
        plt.xticks(new[0], new[1])

    def process_dataset(df, crossref):
        def crossreference(value, col):
            if not pd.isna(value):
                new = crossref[col].iloc[int(value)]
                return new

        def split_frames(text):
            if not pd.isna(text):
                idx = text.find(":")
                subframes.append(text[idx+1:])
                return text[:idx]

        rows_to_drop = np.concatenate([df.loc[np.unique(np.where(pd.isnull(df))[0])].index, df[df["D/H"] == 2].index])
        df.drop(rows_to_drop, axis=0, inplace=True)
        df.drop_duplicates(inplace=True)

        for col in ["time", "source", "frame", "immigrant category", "keyword"]:
            df[col] = df[col].apply(crossreference, col=col)

        subframes = []
        df["frame"] = df["frame"].apply(split_frames)
        df.insert(4, "subframe",pd.Series(subframes))
        df["time"] = pd.to_datetime(df["time"], format="%b-%y")
        df.rename(columns={"immigrant category": "country"}, inplace=True)
        df["source"] = df.pop("source")
        df["link"] = df.pop("link")
        df["keyword"] = df.pop("keyword")
        return df

    def round_to_sf(num, n=2):
        count = 0
        while round(num, count) == 0:
            count += 1
        return round(num, count+n)

## Set up VAD object

VAD = VAD_analysis()