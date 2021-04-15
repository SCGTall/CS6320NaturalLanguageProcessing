# -*- coding: utf-8 -*-
import json
import sys

FOLDER_PREFIX = "data/"
MISINFORMATION_TARGETS = "misinformation_targets.json"
misinformation_targets = {}  # misinformation targets
TRAIN_CANDIDATES = "X1_train_candidates.json"
train_candidates = []  # train candidates
TRAIN_TWEETS = "train_tweets.json"
train_tweets = {}  # train tweets
TRAIN_LABELS = "train_labels.json"
train_labels = {}  # train labels
EXAMPLE_TRAIN_LABELS = "example_train_labels.json"
example_train_labels = {}  # example labels
example_train_candidates = []  # example candidates
annotates = {
    "1": "agree",
    "2": "disagree",
    "3": "no_stance",
    "4": "not_relevant"
}

if len(sys.argv) == 1:  # use the default path
    pass
elif len(sys.argv) == 2:  # set candidates
    TRAIN_CANDIDATES = sys.argv[1]
elif len(sys.argv) == 3:  # set candidates and output train_labels
    TRAIN_CANDIDATES = sys.argv[1]
    TRAIN_LABELS = sys.argv[2]
else:
    print("Try to use input in below forms:")
    print("python3 annotator.py")
    print("python3 annotator.py <candidates>")
    print("python3 annotator.py <candidates> <train_labels>")
    sys.exit(1)

# load json files
# load misinformation targets
with open(FOLDER_PREFIX + MISINFORMATION_TARGETS, mode='r', encoding='utf-8') as f:
    misinformation_targets = json.loads(f.read())

# load train tweets (all)
with open(FOLDER_PREFIX + TRAIN_TWEETS, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        tweet = json.loads(line)
        train_tweets[tweet["id"]] = tweet
        line = f.readline()

# load train labels (history)
try:
    with open(FOLDER_PREFIX + TRAIN_LABELS, mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            label = json.loads(line)
            id = label["tweet_id"]
            mid = label["m_id"]
            if id not in train_labels.keys():
                train_labels[id] = {}
            train_labels[id][mid] = label
            line = f.readline()
except FileNotFoundError:
    pass  # TRAIN_LABELS does not exit in folder

# also load example train labels
with open(FOLDER_PREFIX + EXAMPLE_TRAIN_LABELS, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        label = json.loads(line)
        id = label["tweet_id"]
        mid = label["m_id"]
        example_train_candidates.append((id, mid))
        if id not in example_train_labels.keys():
            example_train_labels[id] = {}
        example_train_labels[id][mid] = label
        line = f.readline()

# load train candidates
with open(FOLDER_PREFIX + TRAIN_CANDIDATES, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        candidate = json.loads(line)
        id = candidate["tweet_id"]
        mid = candidate["m_id"]
        train_candidates.append((id, mid))
        line = f.readline()


def compare_func(a: tuple, b: tuple):
    if a[1] < b[1]:
        return -1
    if a[1] == b[1]:
        return 0
    else:
        return 1


def label_candidates(misinfos: dict, tweets: dict, labels: dict, candidates: list, use_filter: bool):
    sorted = candidates.copy()
    sorted.sort(key=lambda x: x[1])
    for (i, v) in enumerate(sorted):
        print("Candidate #" + str(i))
        (id, mid) = v
        label_exist = (id in labels.keys() and mid in labels[id].keys())
        if use_filter and label_exist:
            continue

        m = misinfos[mid]
        tt = tweets[id]
        dic = {
            "tweet_id": id,
            "text": tt["text"],
            "m_id": mid,
            "title": m["title"],
            "m_text": m["text"],
            "m_label": labels[id][mid]["m_label"] if label_exist else "?"
        }
        print_dictionary(dic)
        print('1 ->   "agree";        2 ->        "disagree";')
        print('3 ->   "no_stance";    4 ->        "not_relevant";')
        print('0 ->   Skip this;      "return" -> Return to main menu.')
        print("Annotate:")
        cmd = input()
        if cmd == "return":
            break
        elif cmd == "0":
            continue
        else:
            if id not in labels.keys():
                labels[id] = {}
            labels[id][mid] = {"tweet_id": id, "m_id": mid, "m_text": annotates[cmd]}
            print("Annotate as: " + annotates[cmd] + "\n")


def check_example_labels(misinfos: dict, tweets: dict, labels: dict, candidates: list):
    sorted = candidates.copy()
    sorted.sort(key=lambda x: x[1])
    for (i, v) in enumerate(sorted):
        print("Example #" + str(i))
        (id, mid) = v
        m = misinfos[mid]
        tt = tweets[id]
        lb = labels[id][mid]
        dic = {
            "tweet_id": id,
            "text": tt["text"],
            "m_id": mid,
            "title": m["title"],
            "m_text": m["text"],
            "m_label": lb["m_label"]
            }
        print_dictionary(dic)
        print('Press anything except "return" to check next example.')
        cmd = input()
        if cmd == "return":
            break
    print("No more examples.")


def print_dictionary(dic: dict):
    print("{")
    for (key, value) in dic.items():
        s = '    "' + key + '": '
        splited = ('"' + value + '",').split()
        for word in splited:
            if (len(s) + len(word)) > 120:
                print(s)
                s = "        " + word
            else:
                s = s + " " + word
        print(s)
    print("}")


def save_or_not(labels: dict, candidates: list):
    print("Do you want to save your changes? [y/n]")
    while True:
        cmd = input()
        if cmd == "n":
            print("Give up all changes.")
            break
        if cmd != "y":
            continue
        # write back results to train_labels.json
        with open(FOLDER_PREFIX + TRAIN_LABELS, mode='x', encoding='utf-8') as f:
            for (id, mid) in candidates:
                line = json.dumps(labels[id][mid]) + "\n"
                f.write(line)
        print("Save all changes.")


def main(m_dict, t_dict, l_dict, c_list, el_dict, ec_list):
    print("Welcome.")
    while True:
        print("Main menu:")
        print("1 ->      Label all candidates.")
        print("2 ->      Label all unlabeled candidates.")
        print("3 ->      Check given example train labels.")
        print('"exit" -> Exit.')
        cmd = input()
        if cmd == "exit":
            break
        elif cmd == "1":
            print("1 -> Label all candidates.")
            label_candidates(m_dict, t_dict, l_dict, c_list, False)
        elif cmd == "2":
            print("2 -> Label all unlabeled candidates.")
            label_candidates(m_dict, t_dict, l_dict, c_list, True)
        elif cmd == "3":
            print("3 -> Check given example train labels.")
            check_example_labels(m_dict, t_dict, el_dict, ec_list)
    save_or_not(l_dict, c_list)


# main
main(misinformation_targets, train_tweets, train_labels, train_candidates,
     example_train_labels, example_train_candidates)
print("Process exit now.")
