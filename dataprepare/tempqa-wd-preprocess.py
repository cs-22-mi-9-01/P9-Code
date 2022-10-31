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
    print('All simple questions have been written into the file: simple_ques.json.')


def json_to_triple(filename):
    question = np.array([])
    entity = np.array([])
    # time = np.array([])
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

    with open(filename) as f:
        # find the question and entity in json respectively
        # question.shape = (175,), entity.shape = (330,)
        for line in f.readlines():
            # que_type = re.findall('"CATEGORY": "Simple"')
            jsquesion = re.findall('"TEXT": ".*?"', line)
            jsentity = re.findall('"SURFACEFORM": ".*?"', line)
            # jstime = re.findall('".*? in .*?"', line)
            if jsquesion != []:
                question = np.append(question, line)
            if jsentity != []:
                entity = np.append(entity, line)
            # if jstime != []:
            #     time = np.append(time, line)
                # print(time)

        for i in range(len(question)):
            # extract question and entity as a nparray respectively
            question[i] = question[i].split('"')[3]
            # count the nubmer of IPron
            que_pron = np.append(que_pron, [question[i].split( ).pop(0)])
        with open('prepared_data/question_set.txt', 'w') as f2:
            for i in range(len(question)):
                f2.write(question[i] + '\n')
        f2.close()
        # print(question)
        que_pron = np.array(que_pron)
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
    f.close()
        # for i in range(len(entity)):
        #     entity[i] = entity[i].split('"')[3]
        #     if i%2 == 0:
        #         entity_sub = np.append(entity_h, entity[i])
        #     else:
        #         entity_obj = np.append(entity_h, entity[i])
            # print(entity[i])
        
        # for i in range(len(time)):


if __name__ == '__main__':
    # json_to_triple()
    ques_type('dev.json')
    json_to_triple('simple_ques.json')