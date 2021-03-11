# -*- coding: utf-8 -*-
import re
import sys
# Problem 1 Part B
# produce a QA system
# How to use?
# python3 ProblemB.py  >>>>>>  responding with console input
# python3 ProblemB.py queries.txt >>>>>>  read queries from a file

# Q: Who is the world's most famous pharaoh?
# A: King Tut.
# Q: What was the cost for visiting the tomb of King Tut?
# A: Dirt, decay and scratches on the walls.
# Q: How old is King Tut’s crypt?
# A: More than 3,300-year-old.
# Q: When was the crypt sealed?
# A: More than 3,000 years ago.
# Q: What happened to King Tut’s tomb during most of the repairs?
# A: Stayed open.
# Q: What delayed the work on King Tut’s tomb?
# A: Arab Spring.
# Q: What do Egyptian officials hope?
# A: Tourists will return.

# I design below regular expressions to answer questions only within provided article.
# Hence, misunderstandment or illogic might happen in normal acknowledge.
dic = {  # a map for reg: ans
    r"(?=.*who)(?=.*world)(?=.*most famous)(?=.*pharaoh)": "King Tut.",
    r"(?=.*what)(?=.*cost)(?=.*visit)(?=.*[tomb|crypt])": "Dirt, decay and scratches on the walls.",
    r"(?=.*how old)(?=.*[tomb|crypt])": "More than 3,300-year-old.",
    r"(?=.*when)(?=.*[tomb|crypt])(?=.*seal)": "More than 3,000 years ago.",
    r"(?=.*what)(?=.*happen)(?=.*[tomb|crypt])(?=.*during)(?=.*repair)": "Stayed open.",
    r"(?=.*what)(?=.*delay)(?=.*work)(?=.*[tomb|crypt])": "Arab Spring.",
    r"(?=.*what)(?=.*egyptian officials)(?=.*hope)": "Tourists will return.",
}

# use all regular expressions to match the input query
def query2answer(query):
    for key in dic.keys():
        mat = re.match(key, query)
        if (mat is not None):
            return dic[key]
    return "Well, this is beyond my field."

if len(sys.argv) == 1:  # respond to console input
    print("Welcome. Type your questions line by line. Type 'FINISH' to quit.")
    while True:
        inp = input("")
        if inp == "FINISH":
            break
        ans = query2answer(inp.lower())
        print(ans)
    print("\nQuit after type 'FINISH'.")

elif len(sys.argv) == 2:  # read queries from file
    try:
        dir = sys.argv[1]
        f = open(dir)
        print("Reading queries from " + dir + "\n")
        line = f.readline()
        while line:
            print(line.replace('\n', ''))  # remove \n after each line
            ans = query2answer(line.lower())
            print(ans)
            line = f.readline()
        print("\nQuit after answer all queries.")
    except IOError:
        print("File is unaccessible.")

else:
    print("Unvalid input. Quit immediately.")
    sys.exit()
