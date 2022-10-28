from calendar import c
import re
from collections import Counter
import numpy as np


def json_to_triple():
    question = np.array([])
    que_pron = np.array([])
    with open('tempqa-wd/data/dev.json') as f:
        # find the question in json
        for line in f.readlines():
            line = re.findall('"TEXT": ".*?"', line)
            if line != []:
                question = np.append(question, line)

        for i in range(len(question)):
            # extract question as a nparray
            question[i] = question[i].split('"')[3]
            # count the nubmer of IPron
            que_pron = np.append(que_pron, [question[i].split( ).pop(0)])
        que_pron = np.array(que_pron)
        unique, unique_counts = np.unique(que_pron, return_counts=True)
        print(unique, unique_counts)
        

if __name__ == '__main__':
    json_to_triple()