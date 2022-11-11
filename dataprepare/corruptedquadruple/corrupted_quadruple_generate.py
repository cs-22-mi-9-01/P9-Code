import numpy as np
import pandas as pd
import csv
import re
import json
import os

# convert txt into csv
# filename: test.txt
def txt_to_csv(filename):
    csv_file = filename.split('/')[-1].split('.')[0] + '.csv'
    with open(csv_file, 'w', encoding='utf-8') as f1:
        writer = csv.writer(f1, delimiter='\t')
        with open(filename, encoding='utf-8') as f2:
            for line in f2:
                line = re.split("[\t\n]", line)
                writer.writerow(line)
    print(f"'{filename}' has been converted to '{csv_file}'.")


# generate corrupted quadruple and save it as csv file
# filename: test.csv
# cor_csv_file: corrupted_quadruple_test.csv
def generate_corrupted_quadruple(filename):
    with open(filename, encoding='utf-8') as f:
        cor_csv_file = 'corrupted_quadruple_' + filename.split('/')[-1]
        with open(cor_csv_file, 'a', encoding='utf-8') as f2:
            f2.write("HEAD\tRELATION\tTAIL\tTIME\tANSWER" + '\n')
            for line in f:
                list = line.strip('\n').split('\t')
                if list[-1] == '':
                    list = list[0:-1]
                else:
                    pass
                a = np.array(list)
                for i in range(4):
                    if i == 0:
                        co_qu = np.array([0, a[1], a[2], a[3], a[0]])
                        f2.write('\t'.join(co_qu) + '\n')
                    elif i == 1:
                        co_qu = np.array([a[0], 0, a[2], a[3], a[1]])
                        f2.write('\t'.join(co_qu) + '\n')
                    elif i == 2:
                        co_qu = np.array([a[0], a[1], 0, a[3], a[2]])
                        f2.write('\t'.join(co_qu) + '\n')
                    elif i == 3:
                        co_qu = np.array([a[0], a[1], a[2], 0, a[3]])
                        f2.write('\t'.join(co_qu) + '\n')
    print("Corrupted quadruples are generated, and saved as 'corrupted_quad.csv'.")


# add ID for each fact
def add_fact_id(filename):
    data = pd.read_csv(filename, sep='\t', encoding='utf-8')
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
    data.to_csv(filename, index=False)
    print("FACT_ID has been added.")


 # convert csv into json: [{key: value}, {key: value}...]
def csv_to_json(filename):
    json_file = filename.split('/')[-1].split('.')[0] + '.json'
    with open(json_file, 'w', encoding='utf-8') as js_f:
        js_f.write('[')
        with open(filename, encoding='utf-8') as f2:
            records = csv.DictReader(f2)
            first = True
            for row in records:
                if not first:
                    js_f.write(',')
                first=False
                json.dump(row, js_f, indent=4, ensure_ascii=False)
        js_f.write(']')
    print(f"'{filename}' has been converted to '{json_file}'.")


if __name__ == '__main__':
    txt_to_csv('temp.txt')
    generate_corrupted_quadruple('temp.csv')
    add_fact_id('corrupted_quadruple_temp.csv')
    csv_to_json('corrupted_quadruple_temp.csv')