import re
from collections import Counter

jsquestion = []
que_pron = []
def json_to_triple():
    with open('tempqa-wd/data/dev.json') as f:
        for line in f.readlines():
            line = re.findall('"TEXT": ".*?"', line)
            if line != []:
                jsquestion.append(line)
        for i in range(len(jsquestion)):
            que_pron.append([jsquestion[i][0].split(" ")[1].strip('"')].pop(0))
        que_counter = dict(Counter(que_pron))
        print(f'Interrogative Pronouns:', que_counter)

if __name__ == '__main__':
    json_to_triple()