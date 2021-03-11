# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
# Problem 2 Part A.
# compute the probability under trigram model
TABLES_DIR = "tables"
CORPUS_DIR = "corpus_for_language_models.txt"


def get_corpus():
    corpus = []
    try:
        f = open(CORPUS_DIR)
        print("Reading corpus from " + CORPUS_DIR + "\n")
        line = f.readline()
        while line:
            wordlist = line.lower().split(' ')
            corpus.extend(wordlist)
            line = f.readline()
    except IOError:
        print("File is unaccessible.")
        sys.exit()
    return corpus


def get_unigrams(s):
    wordlist = s.split(' ')
    outp = []
    for i in range(len(wordlist)):
        token = wordlist[i]
        if token not in outp:  # keep unique
            outp.append(wordlist[i])
    return outp


def get_bigrams(s):
    wordlist = s.split(' ')
    outp = []
    for i in range(len(wordlist) - 1):
        token = ' '.join(wordlist[i: i+2])
        if token not in outp:  # keep unique
            outp.append(token)
    return outp


def get_trigrams(s):
    wordlist = s.split(' ')
    outp = []
    for i in range(len(wordlist) - 2):
        token = ' '.join(wordlist[i: i + 3])
        if token not in outp:  # keep unique
            outp.append(token)
    return outp


def get_unigram_counts(c_dt, corpus, c_dict):
    for i in range(len(corpus)):
        w = corpus[i]
        if w in c_dict.keys():  # match w
            ci = c_dict[w]
            c_dt.iloc[0, ci] += 1

    return c_dt


def get_bigram_counts(c_dt, corpus, c_dict):
    for i in range(len(corpus) - 1):
        w1, w2 = corpus[i], corpus[i + 1]
        if w1 in c_dict.keys():  # match w1
            ri = c_dict[w1]
            if w2 in c_dict.keys():  # match w2
                ci = c_dict[w2]
                c_dt.iloc[ri, ci] += 1

    return c_dt


def get_trigram_counts(c_dt, corpus, bigram_counts, c_dict, r_dict):
    for i in range(len(corpus) - 2):
        w1, w2, w3 = corpus[i], corpus[i + 1], corpus[i + 2]
        bg = w1 + ' ' + w2
        if bg in r_dict.keys():  # match w1, w2
            ri = r_dict[bg]
            bigram_counts[ri] += 1
            if w3 in c_dict.keys():  # match w3
                ci = c_dict[w3]
                c_dt.iloc[ri, ci] += 1

    return c_dt


def get_trigram_probabilities(p_dt, bigram_counts, c_dt):
    for ri in range(len(rows)):
        for ci in range(len(columns)):
            n = bigram_counts[ri]
            p_dt.iloc[ri, ci] = c_dt.iloc[ri, ci] / n  # p = c / N

    return p_dt


def count_word(corpus):  # count unique word number in corpus
    dic = {}
    count = 0
    for word in corpus:
        if (word in dic.keys()):
            dic[word] += 1
        else:
            dic[word] = 1
            count += 1

    return count


def laplace_smoothing_for_counts(lsc_dt):
    for ri in range(lsc_dt.shape[0]):  # row
        for ci in range(lsc_dt.shape[1]):  # column
            lsc_dt.iloc[ri, ci] += 1

    return lsc_dt


def laplace_smoothing_for_probabilities(lsp_dt, bigram_counts, c_dt, v):
    for ri in range(c_dt.shape[0]):  # row
        for ci in range(c_dt.shape[1]):  # column
            n = bigram_counts[ri]
            lsp_dt.iloc[ri, ci] = (c_dt.iloc[ri, ci] + 1) / (n + v)  # p = (c + 1) / (N + V)

    return lsp_dt


def laplace_smoothing_for_reconstituted_counts(rec_dt, bigram_counts, c_dt, v):
    for ri in range(c_dt.shape[0]):  # row
        for ci in range(c_dt.shape[1]):  # column
            n = bigram_counts[ri]
            rec_dt.iloc[ri, ci] = (c_dt.iloc[ri, ci] + 1) * n / (n + v)  # p = (c + 1) / (N + V)

    return rec_dt


def katz_backoff_for_probabilities(p_dt, unic_dt, bic_dt, tric_dt, r_dict, c_dict, v, corpus):
    # nested functions:
    def c_func2(x, y):  # C(x, y)
        xi = c_dict[x]
        yi = c_dict[y]
        return bic_dt.iloc[xi, yi]

    def c_func3(x, y, z):  # C(x, y, z)
        ri = r_dict[' '.join([x, y])]
        zi = c_dict[z]
        return tric_dt.iloc[ri, zi]

    def pstar_func1(x):  # P*(x)
        xi = c_dict[x]
        return (unic_dt.iloc[0, xi] + 1) / (len(corpus) + v)

    def pstar_func2(x, y):  # P*(y|x)
        xi = c_dict[x]
        yi = c_dict[y]
        return (bic_dt.iloc[xi, yi] + 1) / (unic_dt.iloc[0, xi] + v)

    def pstar_func3(x, y, z):  # P*(z|x, y)
        xi = c_dict[x]
        yi = c_dict[y]
        ri = r_dict[' '.join([x, y])]
        zi = c_dict[z]
        return (tric_dt.iloc[ri, zi] + 1) / (bic_dt.iloc[xi, yi] + v)

    def alpha_func1(x):  # a(x)
        b1, b2 = 1, 1
        for ci in range(p_dt.shape[1]):
            w = p_dt.columns.values[ci]
            if c_func2(x, w) > 0:
                b1 -= pstar_func2(x, w)
                b2 -= pstar_func1(w)
        return b1 / b2

    def alpha_func2(x, y):  # a(x, y)
        b1, b2 = 1, 1
        for ci in range(p_dt.shape[1]):
            w = p_dt.columns.values[ci]
            if c_func2(x, w) > 0:
                b1 -= pstar_func3(x, y, w)
                b2 -= pstar_func2(y, w)
        return b1 / b2

    trigram_times = 0
    unigram_times = 0
    for ri in range(p_dt.shape[0]):  # row
        for ci in range(p_dt.shape[1]):  # column
            [x, y] = p_dt.index.values[ri].split(' ')
            z = p_dt.columns.values[ci]
            # by definition
            if c_func3(x, y, z) > 0:
                p_dt.iloc[ri, ci] = pstar_func3(x, y, z)
                trigram_times += 1  # compute trigram probability here
            elif c_func2(x, y) > 0:
                if c_func2(y, z) > 0:
                    p_dt.iloc[ri, ci] = alpha_func2(x, y) * pstar_func2(y, z)
                else:
                    p_dt.iloc[ri, ci] = alpha_func2(x, y) * alpha_func1(y) * pstar_func1(z)
                    unigram_times += 1  # compute unigram probability here
            else:
                p_dt.iloc[ri, ci] = pstar_func1(z)
                unigram_times += 1  # compute unigram probability here

    return (p_dt, trigram_times, unigram_times)


def get_total_trigram_probability(dt):  # product of a straight line from top-left to botton-right
    log_sum = 0  # use log space
    flag = True  # never meet p = 0
    for i in range(dt.shape[1] - 2):  # column
        p = dt.iloc[i, i + 2]
        if p <= 0:
            flag = False
            break
        log_sum += np.log(p)
    if flag:
        return np.exp(log_sum)
    else:
        return 0.0


# main
corpus = get_corpus()  # prepare corpus from file
word_num = count_word(corpus)
# prepare directory to save tables
if not os.path.exists(TABLES_DIR):
    os.makedirs(TABLES_DIR)
# compute the probabilities of S1 and S2 by function: trigram
S1 = "Sales of the company to return to normalcy."
S2 = "The new products and services contributed to increase revenue."
sentences = [S1, S2]
sentence_names = ["S1", "S2"]
for si in range(len(sentences)):
    print("\n" + sentence_names[si] + ": " + sentences[si])
    pretreated = sentences[si][:-1].lower() + " ."  # relax s by adding a space before .
    # step 1 -- compute the trigrams
    trigrams = get_trigrams(pretreated)
    print("\nCompute trigrams:")
    print(trigrams)
    # step 2 -- construct tables with unsmoothed trigram counts and probabilities
    columns = get_unigrams(pretreated)
    c_dict = {}  # hash table to save time on searching
    for (i, c) in enumerate(columns):
        c_dict[c] = i
    rows = get_bigrams(pretreated)  # bigram
    r_dict = {}  # hash table to save time on searching
    for (i, r) in enumerate(rows):
        r_dict[r] = i
    bigram_counts = [0] * len(rows)  # used for saving bigram counts for probability
    c_dt = pd.DataFrame(data=np.zeros(shape=[len(rows), len(columns)], dtype=int),
                        index=rows, columns=columns)  # trigram counts
    p_dt = pd.DataFrame(data=np.zeros(shape=[len(rows), len(columns)]),
                        index=rows, columns=columns)  # trigram probabilities
    # traverse corpus for counts and save
    c_dt = get_trigram_counts(c_dt, corpus, bigram_counts, c_dict, r_dict)
    print("\nTrigram counts:")
    print(c_dt)
    c_dt.to_csv(TABLES_DIR + "/" + sentence_names[si] + "_trigram_counts.csv")
    # use counts to calculate probabilities and save
    p_dt = get_trigram_probabilities(p_dt, bigram_counts, c_dt)
    print(p_dt)
    print("\nTrigram probabilities:")
    p_dt.to_csv(TABLES_DIR + "/" + sentence_names[si] + "_trigram_probabilities.csv")
    # step 3 -- construct tables with Laplace-smoothed counts probabilities and re-constituted counts
    # for pandas dataFrame.copy(), default deep=True
    lsc_dt = c_dt.copy()  # Laplace-smoothing trigram counts
    lsp_dt = pd.DataFrame(data=np.zeros(shape=[len(rows), len(columns)]),
                          index=rows, columns=columns)  # Laplace-smoothing trigram probabilities
    lsrec_dt = pd.DataFrame(data=np.zeros(shape=[len(rows), len(columns)]),
                            index=rows, columns=columns)  # Laplace-smoothing re-constituted trigram counts
    # add one to each count for laplace smoothing
    lsc_dt = laplace_smoothing_for_counts(lsc_dt)
    print("\nLaplace-smoothed trigram counts:")
    print(lsc_dt)
    lsc_dt.to_csv(TABLES_DIR + "/" + sentence_names[si] + "_Laplace_smoothed_counts.csv")
    # add one smoothing for probabilities
    lsp_dt = laplace_smoothing_for_probabilities(lsp_dt, bigram_counts, lsc_dt, word_num)
    print("\nLaplace-smoothed trigram probabilities:")
    print(lsp_dt)
    lsp_dt.to_csv(TABLES_DIR + "/" + sentence_names[si] + "_Laplace_smoothed_probabilities.csv")
    # calculate adjusted count
    lsrec_dt = laplace_smoothing_for_reconstituted_counts(lsrec_dt, bigram_counts, lsc_dt, word_num)
    print("\nLaplace-smoothed re-constituted trigram counts:")
    print(lsrec_dt)
    lsrec_dt.to_csv(TABLES_DIR + "/" + sentence_names[si] + "_Laplace_smoothed_reconstituted_counts.csv")
    # step 4 -- construct tables with Katz back-off smoothed probabilities and analyze
    unic_dt = pd.DataFrame(data=np.zeros(shape=[1, len(columns)], dtype=int),
                        index=[''], columns=columns)  # unigram counts
    bic_dt = pd.DataFrame(data=np.zeros(shape=[len(columns), len(columns)], dtype=int),
                           index=columns, columns=columns)  # bigram counts
    kbp_dt = pd.DataFrame(data=np.zeros(shape=[len(rows), len(columns)]),
                        index=rows, columns=columns)  # Katz back-off smoothed trigram probabilities
    unic_dt = get_unigram_counts(unic_dt, corpus, c_dict)
    bic_dt = get_bigram_counts(bic_dt, corpus, c_dict)
    tric_dt = c_dt.copy()  # trigram counts, already get in step 2
    (kbp_dt, trigram_times, unigram_times) = katz_backoff_for_probabilities(kbp_dt, unic_dt, bic_dt, tric_dt,
                                                                            r_dict, c_dict, word_num, corpus)
    print("\nKatz back-off smoothed trigram probabilities:")
    print(kbp_dt)
    kbp_dt.to_csv(TABLES_DIR + "/" + sentence_names[si] + "_Katz_backoff_smoothed_probabilities.csv")
    print("Compute trigram probabilities " + str(trigram_times) + " times.")
    print("Compute unigram probabilities " + str(unigram_times) + " times.")
    # step 5 -- compute three kind of total probabilities
    print("\nTotal probabilities:")
    total_p = get_total_trigram_probability(p_dt)
    print("Without smoothing:" + str(total_p))
    total_p_ls = get_total_trigram_probability(lsp_dt)
    print("Laplace smoothing:" + str(total_p_ls))
    total_p_kb = get_total_trigram_probability(kbp_dt)
    print("Katz back-off smoothing:" + str(total_p_kb))
print("\nExit after finish all work.")
