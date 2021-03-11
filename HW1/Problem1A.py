# -*- coding: utf-8 -*-
import re
# Problem 1 Part A.
# check the regular expression with library re.
print("Language1:")
reg1 = r"(\b[A-Za-z]+\b)\s\1"
print("Regular Expression: /" + reg1 + "/")
testStrings1 = [
    "Humbert Humbert",
    "the the",
    "the bug",
    "the big bug",
    "the bug the",
    "thethe",
]
# traverse all strings and print results for Language 1
for s in testStrings1:
    print("String: " + s)
    res = re.search(reg1, s)
    print("Result: " + str(res is not None))

print("Language2:")
reg2 = r"(?=.*hedge)(?=.*fund)"
print("Regular Expression: /" + reg2 + "/")
testStrings2 = [
    "hedge fund",
    "fund hedge",
    "funds",
    "funds hedge",
]
# traverse all strings and print results for Language 2
for s in testStrings2:
    print("String: " + s)
    res = re.search(reg2, s)
    print("Result: " + str(res is not None))
