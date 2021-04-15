# -*- coding: utf-8 -*-
import sys
import numpy as np

DEBUGGING = False

pos_tag_major = ["NP", "VP", "S", "AP", "PP", "INF-VP", "CC"]


class Node(object):
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


def find_all_parse(parses, str_bp, gramms, table, sent, nd, r, c, indent=""):
    # print(indent, "[node]", nd.value, r, c)
    indent += "\t"

    for v in set(nd.value):
        if DEBUGGING:
            print(indent, v)
        # major POS tag
        if v in pos_tag_major:
            if DEBUGGING:
                print("before\n", str_bp)
            for k in range(r, c):
                str_bp[k] = v
            if DEBUGGING:
                print("after\n", str_bp)

        for j in range(1, c):
            for v1 in table[r, j]:
                for v2 in table[j, c]:
                    cand = v + " " + v1 + " " + v2
                    # print(indent, cand)
                    for g in gramms:
                        if cand == g:
                            nd.add_child(Node([v1]))
                            # print(indent, nd.children[-1].value)
                            find_all_parse(parses, str_bp, gramms, table, sent, nd.children[-1], r, j, indent)
                            nd.add_child(Node([v2]))
                            # print(indent, nd.children[-1].value)
                            find_all_parse(parses, str_bp, gramms, table, sent, nd.children[-1], j, c, indent)

    # output bracketed structure parse
    if r == table.shape[0] - 1 and c == table.shape[1] - 1:
        if DEBUGGING:
            print(str_bp)

        output = ""
        tag_cur = ""

        for i in range(len(str_bp)):
            if str_bp[i] != tag_cur:
                if tag_cur != "":
                    output = output + "] "

                output = output + "[" + str_bp[i] + " "
                tag_cur = str_bp[i]
            else:
                output = output + " "

            output = output + sent[i]

        output = output + "]"
        if DEBUGGING:
            print(output)
        parses += [output]
    return


def get_bracketed_parse(gramms, sent, table):
    # concatenate each grammar into one string for comparison
    grammars = []
    for g in gramms:
        grammars += [" ".join(s for s in g)]
    # print(grammars)

    # syntactic constituent
    root = Node(table[0, -1])
    # output of all bracketed structure parses
    parses = []
    find_all_parse(parses, [""] * len(sent), grammars, table, sent, root, 0, len(table))

    # print("Bracketed structure parses:")
    # for p in set(parses):
    #     print(p)
    # print("Num of parses:", len(parses))

    return parses


def read_files(file_cnf, file_sents):
    with open(file_cnf) as textFile:
        lines = [line.split() for line in textFile]

    grammars = np.array(lines, dtype="object")

    with open(file_sents) as textFile:
        lines = [line.split() for line in textFile]
    sentences = np.array(lines, dtype="object")

    return (grammars, sentences)


def CKY_parsing(grams, sents):
    outputs = []
    # go through every sentences
    for s in sents:
        dict_s = dict(sent=s)

        if DEBUGGING:
            print("S:", s)
        num_word = len(s)
        # CKY parsing table, add 1 in col to unify the format
        table = np.empty((num_word, num_word + 1), dtype=object)
        table.fill([])

        for j in range(1, num_word + 1):
            # get postag of word
            for g in grams:
                # check original word and lowercase one in case for the first word and proper noun
                if s[j - 1] in g or s[j - 1].lower() in g:
                    table[j - 1][j] = table[j - 1][j] + [g[0]]

            if DEBUGGING:
                print(s[j - 1], table[j - 1][j])

            for i in reversed(range(0, j)):
                for k in range(i + 1, j - 1 + 1):
                    for g in grams:
                        # find B C in grammars
                        for b in table[i, k]:
                            for c in table[k, j]:
                                if b == g[1] and c == g[2]:
                                    table[i][j] = table[i][j] + [g[0]]
                                    if DEBUGGING:
                                        print(i, k, g, table[i][j])
            if DEBUGGING:
                print("line", j, ":", table[:, j])

        if DEBUGGING:
            print("table:\n", table)

        dict_s["table"] = table
        # get bracketed structure parse
        parses = get_bracketed_parse(grams, s, table)
        dict_s["num_of_parse"] = len(parses)
        dict_s["bsp"] = list(set(parses))

        outputs.append(dict_s)

    return outputs



if len(sys.argv) != 4:
    print("usage: hw2_CKYparser.py <grammar_filename> <sentence_filename> <output_filename>")
    sys.exit()

print("Processing...")
# debug use. easy to check output of array
np.set_printoptions(linewidth=200)

# file name of Chomsky normal form (CNF), input sentences, and output
file_cnf = sys.argv[1]
file_sents = sys.argv[2]
file_output = sys.argv[3]

# read CNF and input
grammars, sentences = read_files(file_cnf, file_sents)

# CKY parsing
# outputs: list of dictionary with keys: sent, table, num_of_parse, bsp
outputs = CKY_parsing(grammars, sentences)
f_table = open("p1_table.txt", 'w')  # save for check
f = open(file_output, 'w')

for (index, dict_s) in enumerate(outputs):
    s_index = "S" + str(index + 1)
    # Sentence
    print("\n" + s_index + ": ", dict_s["sent"])
    f.write("--" + s_index + "--\n")
    np.savetxt(f, [" ".join(dict_s["sent"])], fmt="%s", header="Sentence")
    # Bracketed structure parses
    print("Bracketed structure parses:")
    for p in dict_s["bsp"]:
        print("\t", p)
    np.savetxt(f, dict_s["bsp"], fmt="%s", header="Bracketed structure parses")
    # Num of parses
    print("Num of parses:", dict_s["num_of_parse"])
    np.savetxt(f, [dict_s["num_of_parse"]], fmt="%d", header="Num of parses")
    # save for check
    np.savetxt(f_table, dict_s["table"], fmt="%s", header=" ".join(dict_s["sent"]))
    f.write("\n")
    f_table.write("\n")

f_table.close()
f.close()

print("\nFinish and exit.")
