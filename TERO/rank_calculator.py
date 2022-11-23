import numpy as np
import torch
import datetime
from datetime import date
import TERO_model as KGE
from Dataset import KnowledgeGraph
from Dataset_YG import KnowledgeGraphYG

def mean_rank(rank):
    m_r = 0
    N = len(rank)
    for i in rank:
        m_r = m_r + i / N

    return m_r


def mrr(rank):
    mrr = 0
    N = len(rank)
    for i in rank:
        mrr = mrr + 1 / i / N

    return mrr


def hit_N(rank, N):
    hit = 0
    for i in rank:
        if i <= N:
            hit = hit + 1

    hit = hit / len(rank)

    return hit

class RankCalculator:
    # if ((epoch+1)//min_epoch>epoch//min_epoch and epoch < max_epoch) :
    if task == 'LinkPrediction':
        rank = model.rank_left(validation_pos,kg.validation_facts,kg,timedisc,rev_set=rev_set)
        rank_right = model.rank_right(validation_pos,kg.validation_facts,kg,timedisc,rev_set=rev_set)
        rank = rank + rank_right
    #     else:
    #         rank = model.timepred(validation_pos)

    m_rank = mean_rank(rank)
    mean_rr = mrr(rank)
    hit_1 = hit_N(rank, 1)
    hit_3 = hit_N(rank, 3)
    hit_5 = hit_N(rank, 5)
    hit_10 = hit_N(rank, 10)
    print('validation results:')
    print('Mean Rank: {:.0f}'.format(m_rank))
    print('Mean RR: {:.4f}'.format(mean_rr))
    print('Hit@1: {:.4f}'.format(hit_1))
    print('Hit@3: {:.4f}'.format(hit_3))
    print('Hit@5: {:.4f}'.format(hit_5))
    print('Hit@10: {:.4f}'.format(hit_10))
    f = open(os.path.join(path, 'result{:.0f}.txt'.format(epoch)), 'w')
    f.write('Mean Rank: {:.0f}\n'.format(m_rank))
    f.write('Mean RR: {:.4f}\n'.format(mean_rr))
    f.write('Hit@1: {:.4f}\n'.format(hit_1))
    f.write('Hit@3: {:.4f}\n'.format(hit_3))
    f.write('Hit@5: {:.4f}\n'.format(hit_5))
    f.write('Hit@10: {:.4f}\n'.format(hit_10))
    for loss in losses:
        f.write(str(loss))
        f.write('\n')
    f.close()
    if mean_rr < mrr_std and patience<3:
        patience+=1
    elif (mean_rr < mrr_std and patience>=3) or epoch==max_epoch-1:
        if epoch == max_epoch-1:
            torch.save(model.state_dict(), os.path.join(path, 'params.pkl'))
        model.load_state_dict(torch.load(os.path.join(path,'params.pkl')))
        if task == 'LinkPrediction':
            rank = model.rank_left(test_pos,kg.test_facts,kg,timedisc,rev_set=rev_set)
            rank_right = model.rank_right(test_pos,kg.test_facts,kg,timedisc,rev_set=rev_set)
            rank = rank + rank_right
        else:
            rank = model.timepred(test_pos)


        m_rank = mean_rank(rank)
        mean_rr = mrr(rank)
        hit_1 = hit_N(rank, 1)
        hit_3 = hit_N(rank, 3)
        hit_5 = hit_N(rank, 5)
        hit_10 = hit_N(rank, 10)
        print('test result:')
        print('Mean Rank: {:.0f}'.format(m_rank))
        print('Mean RR: {:.4f}'.format(mean_rr))
        print('Hit@1: {:.4f}'.format(hit_1))
        print('Hit@3: {:.4f}'.format(hit_3))
        print('Hit@5: {:.4f}'.format(hit_5))
        print('Hit@10: {:.4f}'.format(hit_10))
        if epoch == max_epoch-1:
            f = open(os.path.join(path, 'test_result{:.0f}.txt'.format(epoch)), 'w')
        else:
            f = open(os.path.join(path, 'test_result{:.0f}.txt'.format(epoch)), 'w')
        f.write('Mean Rank: {:.0f}\n'.format(m_rank))
        f.write('Mean RR: {:.4f}\n'.format(mean_rr))
        f.write('Hit@1: {:.4f}\n'.format(hit_1))
        f.write('Hit@3: {:.4f}\n'.format(hit_3))
        f.write('Hit@5: {:.4f}\n'.format(hit_5))
        f.write('Hit@10: {:.4f}\n'.format(hit_10))
        for loss in losses:
            f.write(str(loss))
            f.write('\n')
        f.close()
