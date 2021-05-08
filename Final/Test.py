# -*- coding: utf-8 -*-
bag = ["blue", "river", "ice", "mountain", "sky", "sun", "water", "weather", "wind", "zen"]
article = """All clouds are made up of basically the same thing water droplets or different ice crystal that float in the sky But all clouds look a little bit different from one another and sometimes these differences can help us predict a change in the weather"""
words = article.lower().split()
count = [0] * len(bag)
bag_dic = {v: i for (i, v) in enumerate(bag)}
for word in words:
    if word in bag_dic.keys():
        count[bag_dic[word]] = 1

print(count)
