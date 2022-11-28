import re
import numpy as np
import json


def ques_type(filename):
    ques_simple = np.array([])
    with open(f'tempqa-wd/data/{filename}') as file:
        res = json.load(file)
        res = np.array(res)
        for i in range(len(res)):
            if res[i].get('CATEGORY') == 'Simple':
                ques_simple = np.append(ques_simple, res[i])
        file.close()
    qsimple_lis = ques_simple.tolist()
    qsimple_str = json.dumps(qsimple_lis)
    qsimple_dic = json.loads(qsimple_str)

    with open('simple_ques.json', 'w') as f:
        json.dump(qsimple_dic, f, indent=4)
    print('All simple questions (in json) have been written into the file: simple_ques.json.')


def ques_extract(filename):
    question = np.array([])

    with open(filename) as f:
    # find the question and entity in json respectively
        # question.shape = (175,), entity.shape = (330,)
        for line in f.readlines():
            jsquesion = re.findall('"TEXT": ".*?"', line)
            if jsquesion != []:
                question = np.append(question, line)
        for i in range(len(question)): 
            question[i] = question[i].split('"')[3]
    # Put all questions into a file
        with open('prepared_data/question_set.txt', 'w') as f2:
            for i in range(len(question)):
                f2.write(question[i] + '\n')
        f2.close()
    f.close()
    print("All simple questions (in natural language) have been written into the file: question_set.txt. ")
    return question


def ques_classify():
    que_pron = np.array([])

    # Create file for interrogative word
    what_file = open('prepared_data/what.txt')
    what_file.close()
    when_file = open('prepared_data/when.txt')
    when_file.close()
    where_file = open('prepared_data/where.txt')
    where_file.close()
    which_file = open('prepared_data/which.txt')
    which_file.close()
    who_file = open('prepared_data/who.txt')
    who_file.close()

    # Classify all questions into different file by interrogative words, such as 'which'.
    question = ques_extract('simple_ques.json')
    for i in range(len(question)):
        # count the nubmer of IPron
        que_pron = np.append(que_pron, [question[i].split( ).pop(0)])
    for i in range(len(que_pron)):
        print(que_pron)
        if que_pron[i] == "what":
            with open('prepared_data/what.txt', 'a') as which_file:
                which_file.write(question[i] + '\n')
        elif que_pron[i] == "when":
            with open('prepared_data/when.txt', 'a') as which_file:
                which_file.write(question[i] + '\n')
        elif que_pron[i] == "where":
            with open('prepared_data/where.txt', 'a') as which_file:
                which_file.write(question[i] + '\n')
        elif que_pron[i] == "which":
            with open('prepared_data/which.txt', 'a') as which_file:
                which_file.write(question[i] + '\n')
        elif que_pron[i] == "who":
            with open('prepared_data/who.txt', 'a') as which_file:
                which_file.write(question[i] + '\n')
    unique, unique_counts = np.unique(que_pron, return_counts=True)
    print(f'Interrogative Pronouns: ',unique, unique_counts)


def json_to_triple(filename):
    quadruple = []
    entity = np.array([])

    with open(filename) as f:
        for line in f.readlines():
                jsentity = re.findall('"SURFACEFORM": ".*?"', line)
                # jstime = re.findall('".*? in .*?"', line)
                if jsentity != []:
                    entity = np.append(entity, line)

                # if jstime != []:
                #     time = np.append(time, line)
                    # print(time)

        for i in range(len(entity)): 
            entity[i] = entity[i].split('"')[3]
        # time = np.array([])

        # Build the quadruple
        h_entity = np.array([])
        t_entity = np.array([])
        for i in range(len(entity)):
            if i%2 == 0:
                h_entity = np.append(h_entity, entity[i])
            else:
                t_entity = np.append(t_entity, entity[i])

        for i in range(len(h_entity)):
            quadruple.append([h_entity[i], 'init_relation', t_entity[i], 'init_time'])
        print(quadruple[0])

if __name__ == '__main__':
    # ques_type('dev.json')
    json_to_triple('simple_ques.json')
