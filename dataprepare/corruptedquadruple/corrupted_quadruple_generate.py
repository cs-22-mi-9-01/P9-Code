import numpy as np
import pandas as pd
import csv
import re
import json

# convert txt into csv
def txt_to_csv(filename):
    with open(f"{filename.split('/')[-1].split('.')[0]}.csv", 'w') as f1:
        writer = csv.writer(f1)
        with open(filename) as f2:
            for line in f2:
                line = re.split("['\t','\n']", line)
                writer.writerow(line)

# generate corrupted quadruple and save it as csv file
def generate_corrupted_quadruple(filename):
    with open(filename) as f:
        with open('corrupted_quad.csv', 'a') as f2:
            f2.write('HEAD,RELATION,TAIL,TIME,ANSWER' + '\n')
            for line in f:
                list = line.strip('\n').split(',')
                if list[-1] == '':
                    list = list[0:-1]
                else:
                    pass
                a = np.array(list)
                for i in range(4):
                    if i == 0:
                        co_qu = np.array([0, a[1], a[2], a[3], a[0]])
                        f2.write(','.join(co_qu) + '\n')
                    elif i == 1:
                        co_qu = np.array([a[0], 0, a[2], a[3], a[1]])
                        f2.write(','.join(co_qu) + '\n')
                    elif i == 2:
                        co_qu = np.array([a[0], a[1], 0, a[3], a[2]])
                        f2.write(','.join(co_qu) + '\n')
                    elif i == 3:
                        co_qu = np.array([a[0], a[1], a[2], 0, a[3]])
                        f2.write(','.join(co_qu) + '\n')


def add_title(filename):
    data = pd.read_csv(filename)
    fact_column = []
    num1 = 0
    num2 = 0
    for index, row in data.iterrows():
        if index % 4 == 0:
            num1 += 1
            num2 = 0
        else:
            pass
        fact_column.append(f'FACT_{num1}_{num2}')
        num2 += 1
    data['FACT_ID'] = fact_column
    data = data[['FACT_ID', 'HEAD', 'RELATION', 'TAIL', 'TIME', 'ANSWER']]
    data.to_csv('cc.csv', index=False)


# convert csv into json: [obj, obj,obj...]
def csv_to_json(filename):
    with open('corrupted_quad.json', 'w') as f:
        all_data = pd.read_csv(filename)
        # TODO: write too much rows at once
        a = np.array([])
        for i in range(len(all_data)):
            row = ','.join(all_data.loc[i,:])
            row = "{%s}" % row
            a = np.append(a, row).tolist()
        json.dump(a, f, indent=4)


 # convert csv into json: [{key: value}, {key: value}...]
def csv_to_json_2(filename):
    with open('corrupted_quad.json', 'w') as f:
        with open(filename) as f2:
            records = csv.DictReader(f2)
            a = np.array([])
            for row in records:
                a = np.append(a, row).tolist()
            json.dump(a, f, indent=4)
    # data = pd.read_csv(filename, header=0)
    # # print(data)
    # for line in data.values:
    #     dic = {}
    #     list = line.tolist()
    #     print(list[0])
    #     print(','.join(list[1:]))
    #     # for item, data in zip(list[0], list[1:]):
    #     #     dic[item] = data
    #     #     print(dic)


if __name__ == '__main__':
    # txt_to_csv('temp.txt')
    # generate_corrupted_quadruple('temp.csv')
    # add_title('corrupted_quad.csv')
    # csv_to_json('corrupted_quad.csv')
    csv_to_json_2('cc.csv')