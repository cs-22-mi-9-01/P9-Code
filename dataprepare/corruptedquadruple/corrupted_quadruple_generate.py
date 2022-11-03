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
            for line in f:
                list = line.split(',')
                if list[-1] == '\n':
                    list = list[0:-1]
                else:
                    pass
                print(list)
                a = np.array(list)
                print(len(a))
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
                    else:
                        i = 4

# convert csv into json
def csv_to_json(filename):
    with open('corrupted_quad.json', 'w') as f:
        all_data = pd.read_csv(filename)
        for i in range(len(all_data)):
            row = all_data.loc[i,:].tolist()
            json.dump(row, f, indent=4)

    

if __name__ == '__main__':
    # txt_to_csv('temp.txt')
    generate_corrupted_quadruple('temp.csv')
    # csv_to_json('corrupted_quad.csv')