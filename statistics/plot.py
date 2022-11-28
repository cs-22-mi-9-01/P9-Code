import matplotlib.pyplot as plt
import json
import pandas as pd


def hypothesis_1(*args):
    # Concatenate json file
    files = []
    keys = ['HEAD', 'RELATION', 'TAIL', 'TIME']
    for i in range(len(args)):
        with open(args[i]) as f:
            files.append(dict(json.load(f).items()))
                # 为什么加items才显示值？
    dic = dict(zip(keys, files))
    with open('result/icews14/hypothesis_1.json', 'w') as f:
        json.dump(dic, f, indent=4)
    print(dic)
    
    xAxis = ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE']
    yAxis_HEAD = []
    yAxis_RELATION = []
    yAxis_TAIL = []
    yAxis_TIME = []

    dictionary = json.load(open('result/icews14/hypothesis_1.json', 'r'))
    for key, value in dictionary.items():
        for key1, value1 in value.items():
            if key == 'HEAD':
                yAxis_HEAD.append(value1['MRR'])
            elif key == 'RELATION':
                yAxis_RELATION.append(value1['MRR'])
            elif key == 'TAIL':
                yAxis_TAIL.append(value1['MRR'])
            elif key == 'TIME':
                yAxis_TIME.append(value1['MRR'])
    

    # BAR GRAPH
    fig = plt.figure()
    x = list(range(len(xAxis)))
    width = 0.1
    plt.bar(x, yAxis_HEAD, width=width, label='HEAD', tick_label=xAxis)
    for i in range(len(xAxis)):
        x[i] = x[i] + width
    plt.bar(x, yAxis_RELATION, width=width, label='RELATION', tick_label=xAxis)
    for i in range(len(xAxis)):
        x[i] = x[i] + width
    plt.bar(x, yAxis_TAIL, width=width, label='TAIL', tick_label=xAxis)
    for i in range(len(xAxis)):
        x[i] = x[i] + width
    plt.bar(x, yAxis_TIME, width=width, label='TIME', tick_label=xAxis)
    
    plt.xlabel('MODEL')
    plt.ylabel('MRR')
    plt.title('Hypothesis 1')
    plt.legend()
    fig.savefig("figure/icews14/hypothesis_1.png")


#  Time Series Plot
# TODO: The time is right?
# TODO: Maybe a better plot to represent distribution
def time_distribution(filename, type):
    list_Data = json.load(open(filename, 'r'))
    xAxis = []
    if type == 'num':
        yAxis = []
        for i in range(len(list_Data)):
            for key, value in list_Data[i].items():
                if key == 'TIME':
                    xAxis.append(list_Data[i][key])
                elif key == 'NUM_FACTS':
                    yAxis.append(list_Data[i][key])
        # Draw plot
        fig = plt.figure(figsize=(16,10), dpi= 80)
        df = pd.DataFrame(list(zip(xAxis, yAxis)), columns=['TIME', 'NUM_FACTS'])
        plt.plot('TIME', 'NUM_FACTS', data=df, marker='o' )
        xtick_location = df.index.tolist()[::12]
        xtick_labels = [x[-5:] for x in df.TIME.tolist()[::12]]
        plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Time distribution (2014)", fontsize=22)
        plt.grid(axis='both', alpha=.3)
        plt.xlabel('TIME')
        plt.ylabel('NUM')
        plt.legend()
        plt.show()
        fig.savefig('figure/icews14/time_num_distribution.png')
    elif type == 'mrr':
        yAxis_DE_TransE = []
        yAxis_DE_SimplE = []
        yAxis_DE_DistMult = []
        for i in range(len(list_Data)):
            for key, value in list_Data[i].items():
                if key == 'TIME':
                    xAxis.append(list_Data[i][key])
                elif key == 'MEASURE':
                    for key1, value1 in list_Data[i][key].items():
                        if key1 == 'DE_TransE':
                            yAxis_DE_TransE.append(list_Data[i][key][key1]['MRR'])
                        elif key1 == 'DE_SimplE':
                            yAxis_DE_SimplE.append(list_Data[i][key][key1]['MRR'])
                        elif key1 == 'DE_DistMult':
                            yAxis_DE_DistMult.append(list_Data[i][key][key1]['MRR']) 
        # Draw plot
        fig = plt.figure(figsize=(16,10), dpi= 80)
        df_DE_TransE = pd.DataFrame(list(zip(xAxis, yAxis_DE_TransE)), columns=['TIME', 'DE_TransE'])
        plt.plot('TIME', 'DE_TransE', data=df_DE_TransE, marker='o' )

        df_DE_SimplE = pd.DataFrame(list(zip(xAxis, yAxis_DE_SimplE)), columns=['TIME', 'DE_SimplE'])
        plt.plot('TIME', 'DE_SimplE', data=df_DE_SimplE, marker='o' )

        df_DE_DistMult = pd.DataFrame(list(zip(xAxis, yAxis_DE_DistMult)), columns=['TIME', 'DE_DistMult'])
        plt.plot('TIME', 'DE_DistMult', data=df_DE_DistMult, marker='o' )
        xtick_location = df_DE_TransE.index.tolist()[::12]
        xtick_labels = [x[-5:] for x in df_DE_TransE.TIME.tolist()[::12]]
        plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Time MRR distribution (2014)", fontsize=22)
        plt.grid(axis='both', alpha=.3)
        plt.xlabel('TIME')
        plt.ylabel('MRR')
        plt.legend()
        plt.show()
        fig.savefig('figure/icews14/time_mrr_distribution.png')


def entity_distribution(filename, type):
    list_Data = json.load(open(filename, 'r'))
    lower_bound = 100
    xAxis = []
    if type == 'num':
        yAxis = []
        for i in range(len(list_Data)):
            for key, value in list_Data[i].items():
                if key == 'NUM_FACTS':
                    if list_Data[i][key] > lower_bound:
                        for key1, value1 in list_Data[i].items():
                            if key1 == 'ENTITY':
                                xAxis.append(list_Data[i][key1])
                            elif key1 == 'NUM_FACTS':
                                yAxis.append(list_Data[i][key1])
         # Draw plot
        fig = plt.figure(figsize=(16,10), dpi= 80)
        df = pd.DataFrame(list(zip(xAxis, yAxis)), columns=['ENTITY', 'NUM_FACTS'])
        plt.plot('ENTITY', 'NUM_FACTS', data=df, marker='o' )
        plt.title("Entity distribution", fontsize=22)
        plt.xticks(rotation=90)
        plt.grid(axis='both', alpha=.3)
        plt.legend()
        plt.show()
        fig.savefig('figure/icews14/entity_num_distribution.png')
    elif type == 'mrr':
        yAxis_DE_TransE = []
        yAxis_DE_SimplE = []
        yAxis_DE_DistMult = []
        for i in range(len(list_Data)):
            for key, value in list_Data[i].items():
                if key == 'NUM_FACTS':
                    if list_Data[i][key] > lower_bound:
                        for key1, value1 in list_Data[i].items():
                            if key1 == 'ENTITY':
                                xAxis.append(list_Data[i][key1])
                            elif key1 == 'MEASURE':
                                for key2, value in list_Data[i][key1].items():
                                    if key2 == 'DE_TransE':
                                        yAxis_DE_TransE.append(list_Data[i][key1][key2]['MRR'])
                                    elif key2 == 'DE_SimplE':
                                        yAxis_DE_SimplE.append(list_Data[i][key1][key2]['MRR'])
                                    elif key2 == 'DE_DistMult':
                                        yAxis_DE_DistMult.append(list_Data[i][key1][key2]['MRR'])
        # Draw Plot
        fig = plt.figure(figsize=(16,10), dpi= 80)
        df_DE_TransE = pd.DataFrame(list(zip(xAxis, yAxis_DE_TransE)), columns=['ENTITY', 'DE_TransE'])
        plt.plot('ENTITY', 'DE_TransE', data=df_DE_TransE, marker='o' )

        df_DE_SimplE = pd.DataFrame(list(zip(xAxis, yAxis_DE_SimplE)), columns=['ENTITY', 'DE_SimplE'])
        plt.plot('ENTITY', 'DE_SimplE', data=df_DE_SimplE, marker='o' )

        df_DE_DistMult = pd.DataFrame(list(zip(xAxis, yAxis_DE_DistMult)), columns=['ENTITY', 'DE_DistMult'])
        plt.plot('ENTITY', 'DE_DistMult', data=df_DE_DistMult, marker='o' )

        # Decoration
        plt.title("Entity MRR distribution", fontsize=22)
        plt.xticks(rotation=90)
        plt.grid(axis='both', alpha=.3)
        plt.legend()
        plt.show()
        fig.savefig('figure/icews14/entity_mrr_distribution.png')


def relation_distribution(filename, type):
    list_Data = json.load(open(filename, 'r'))
    lower_bound = 10
    xAxis = []
    if type == 'num':
        yAxis = []
        for i in range(len(list_Data)):
            for key, value in list_Data[i].items():
                if key == 'NUM_FACTS':
                    if list_Data[i][key] > lower_bound:
                        for key1, value1 in list_Data[i].items():
                            if key1 == 'RELATION':
                                xAxis.append(list_Data[i][key1])
                            elif key1 == 'NUM_FACTS':
                                yAxis.append(list_Data[i][key1])
         # Draw plot
        fig = plt.figure(figsize=(16,10), dpi= 80)
        df = pd.DataFrame(list(zip(xAxis, yAxis)), columns=['RELATION', 'NUM_FACTS'])
        plt.plot('RELATION', 'NUM_FACTS', data=df, marker='o' )
        plt.title("Relation distribution", fontsize=22)
        plt.xticks(rotation=90)
        plt.grid(axis='both', alpha=.3)
        plt.legend()
        plt.show()
        fig.savefig('figure/icews14/relation_num_distribution.png')
    elif type == 'mrr':
        yAxis_DE_TransE = []
        yAxis_DE_SimplE = []
        yAxis_DE_DistMult = []
        for i in range(len(list_Data)):
            for key, value in list_Data[i].items():
                if key == 'NUM_FACTS':
                    if list_Data[i][key] > lower_bound:
                        for key1, value1 in list_Data[i].items():
                            if key1 == 'RELATION':
                                xAxis.append(list_Data[i][key1])
                            elif key1 == 'MEASURE':
                                for key2, value in list_Data[i][key1].items():
                                    if key2 == 'DE_TransE':
                                        yAxis_DE_TransE.append(list_Data[i][key1][key2]['MRR'])
                                    elif key2 == 'DE_SimplE':
                                        yAxis_DE_SimplE.append(list_Data[i][key1][key2]['MRR'])
                                    elif key2 == 'DE_DistMult':
                                        yAxis_DE_DistMult.append(list_Data[i][key1][key2]['MRR'])
        # Draw Plot
        fig = plt.figure(figsize=(16,10), dpi= 80)
        df_DE_TransE = pd.DataFrame(list(zip(xAxis, yAxis_DE_TransE)), columns=['RELATION', 'DE_TransE'])
        plt.plot('RELATION', 'DE_TransE', data=df_DE_TransE, marker='o' )

        df_DE_SimplE = pd.DataFrame(list(zip(xAxis, yAxis_DE_SimplE)), columns=['RELATION', 'DE_SimplE'])
        plt.plot('RELATION', 'DE_SimplE', data=df_DE_SimplE, marker='o' )

        df_DE_DistMult = pd.DataFrame(list(zip(xAxis, yAxis_DE_DistMult)), columns=['RELATION', 'DE_DistMult'])
        plt.plot('RELATION', 'DE_DistMult', data=df_DE_DistMult, marker='o' )

        # Decoration
        plt.title("Relation MRR distribution", fontsize=22)
        plt.xticks(rotation=90)
        plt.grid(axis='both', alpha=.3)
        plt.legend()
        plt.show()
        fig.savefig('figure/icews14/relation_mrr_distribution.png')


if __name__ == '__main__':
    hypothesis_1('result/icews14/hypothesis_1_head.json', 'result/icews14/hypothesis_1_relation.json', 'result/icews14/hypothesis_1_tail.json', 'result/icews14/hypothesis_1_time.json')
    # time_distribution('result/icews14/hypothesis_2_time.json', 'mrr')
    # entity_distribution('result/icews14/hypothesis_2_entity.json', 'mrr')
    # relation_distribution('result/icews14/hypothesis_2_relation.json', 'num')
    