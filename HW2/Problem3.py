# -*- coding: utf-8 -*-
import spacy
from spacy import displacy

nlp_sm = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

file = open("sentences.txt")
input = []

for (i, sent) in enumerate(file):
    input.append(("S" + str(i+1), sent))

documents = []
for pair in input:
    print(": ".join(pair))
    sent = pair[1]
    documents.append(nlp_sm(sent))
    documents.append(nlp_trf(sent))
displacy.serve(documents, options={"compact": True}, style='dep')
