# -*- coding: utf-8 -*-
import sys
import re
import numpy as np
import pandas as pd
# Problem 3
# about vector semantics
CORPUS_DIR = "corpus_for_language_models.txt"
NIL_STR = "NIL"
LEFT_LEN = 5
RIGHT_LEN = 5


def get_corpus():
    corpus = []
    reg = r"[0-1a-zA-Z]+"
    try:
        f = open(CORPUS_DIR)
        print("Reading corpus from " + CORPUS_DIR + "\n")
        line = f.readline()
        while line:
            for i in range(max(LEFT_LEN, RIGHT_LEN)):  # make sure that there will always have enough NIL
                corpus.append("NIL")
            wordlist = line.lower().split(' ')
            for word in wordlist:
                res = re.search(reg, word)  # only consider word
                if res is not None:
                    corpus.append(word)
            line = f.readline()
        for i in range(max(LEFT_LEN, RIGHT_LEN)):  # add NIL at the end
            corpus.append("NIL")
    except IOError:
        print("File is unaccessible.")
        sys.exit()
    return corpus


def get_term_context_matrix(corpus, words, contexts, w_dict, c_dict):
    mat = [[0 for j in range(len(contexts))] for i in range(len(words))]
    for index in range(len(corpus)):
        word = corpus[index]
        if word in words:
            window = corpus[index - 5 : index + 6]
            for context in contexts:
                if context in window:
                    mat[w_dict[word]][c_dict[context]] += 1

    return mat


def analyze_term_context_matrix(mat):
    width, height = len(mat[0]), len(mat)
    sum = 0
    w_c = [0] * height
    c_c = [0] * width
    for i in range(height):
        for j in range(width):
            sum += mat[i][j]
            w_c[i] += mat[i][j]
            c_c[j] += mat[i][j]
    return (sum, w_c, c_c)


# main
# part 1, compute the PPMI
print("\nPart 1, compute the PPMI")
corpus = get_corpus()
# init
words = ["chairman", "company"]
contexts = ["said", "of", "board"]
w_dict = {}  # use hash table to save time on searching
for (i, v) in enumerate(words):
    w_dict[v] = i
c_dict = {}
for (i, v) in enumerate(contexts):
    c_dict[v] = i
# get term-context matrix
mat = get_term_context_matrix(corpus, words, contexts, w_dict, c_dict)  # term-context matrix
df = pd.DataFrame(data=np.asarray(mat), index=words, columns=contexts)  # print in a nice way
print("Term-context matrix:")
print(df)
(sum_count, word_counts, context_counts) = analyze_term_context_matrix(mat)  # get the sum of rows, columns and total
print("\nSum = " + str(sum_count))
# compute PPMI
print("\nPPMIs:")
PPMI_tasks = [("chairman", "said"),
         ("chairman", "of"),
         ("company", "board"),
         ("company", "said"),]  # first -> word, second -> context
for task in PPMI_tasks:
    wi = w_dict[task[0]]  # index of word
    ci = c_dict[task[1]]  # index of context
    pxy = mat[wi][ci] / sum_count
    px = word_counts[wi] / sum_count
    py = context_counts[ci] / sum_count
    pmi = np.log2(pxy / (px * py))
    ppmi = max(pmi, 0.0)
    print("PPMI" + str(task) + " = " + str(ppmi))

# part 2, compute the similarity
print("\nPart 2, compute the similarity")
words2 = ["chairman", "company", "sales", "economy"]
contexts2 = ["said", "of", "board"]
w_dict2 = {}  # use hash table to save time on searching
for (i, v) in enumerate(words2):
    w_dict2[v] = i
c_dict2 = {}
for (i, v) in enumerate(contexts2):
    c_dict2[v] = i
# get term-context matrix
mat2 = get_term_context_matrix(corpus, words2, contexts2, w_dict2, c_dict2)  # term-context matrix
df2 = pd.DataFrame(data=np.asarray(mat2), index=words2, columns=contexts2)  # print in a nice way
print("Term-context matrix:")
print(df2)
(sum_count2, word_counts2, context_counts2) = analyze_term_context_matrix(mat2)  # get the sum of rows, columns and total
print("\nSum = " + str(sum_count2))
# compute similarity
print("\nSimilarities:")
similarity_tasks = [("chairman", "company"),
         ("company", "sales"),
         ("company", "economy"),]
for task in similarity_tasks:
    wi1 = w_dict2[task[0]]  # index of word v
    wi2 = w_dict2[task[1]]  # index of word w
    sum_vw = 0
    sum_v2 = 0
    sum_w2 = 0
    for ci in range(len(contexts2)):
        sum_vw += mat2[wi1][ci] * mat2[wi2][ci]
        sum_v2 += mat2[wi1][ci] * mat2[wi1][ci]
        sum_w2 += mat2[wi2][ci] * mat2[wi2][ci]
    cos = sum_vw / (np.sqrt(sum_v2) * np.sqrt(sum_w2))
    print("cos" + str(task) + " = " + str(cos))