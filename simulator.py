# coding: utf-8
__author__ = "GU_ZHIXIANG"
__date__ = '2017/09/21'
'''
テキスト発話と共起するシミュレーションのデータ
を生成するモジュール

'''

import numpy as np
import numpy.random as rd
from math import *
import pdb


def creat_sample(sp, num_ag, num_ob):
    sp = dict.fromkeys(sp, True)
    # print sp
    #   パラメータ設定
    para = ([0., 15., .5, .20],
            [pi, 15., .5, .20],
            [pi / 2, 15., .5, .20],
            [-pi / 2, 15., .5, .20])
    #   リスト設定
    # list_ob = np.array(
    #     (["てれび", 0], ["せんぷうき", 1], ["すくりいん", 2], ["つくえ", 3], ["ぱそこん", 4], ["れえぞおこ", 5], ["いす", 6], ["ほんだな", 7], ["そふぁあ", 8], ["ごみばこ", 9], ["ぺっぱあ", 10], ["べっど", 11], ["すとおぶ", 12], ["ろっかあ", 13], ["はちうえ", 14], ["はしら", 15]))
    # list_ob = list_ob[:num_ob, :]
    # list_cp = np.array((["まえ", 0], ["うしろ", 1], ["ひだり", 2], ["みぎ", 3]))

    list_ob = np.array(
        (["terebi", 0], ["senfuki", 1], ["sukurin", 2], ["tukue", 3], ["pasokon", 4], ["rezoko", 5], ["isu", 6], ["hondana", 7], ["sofa", 8], ["gomibako", 9], ["peppa", 10], ["beddo", 11], ["sutofu", 12], ["rokka", 13], ["hachiue", 14], ["hashira", 15]))
    list_ob = list_ob[:num_ob, :]
    list_cp = np.array((["mae", 0], ["ushiro", 1], ["hidari", 2], ["migi", 3]))

    ref_ob = int([ob[1] for ob in list_ob if ob[0] in sp][0])
    ref_cp = int([cp[1] for cp in list_cp if cp[0] in sp][0])
    # "参照物：",ref_ob, "概念：",ref_cp
    list_index_ob = list(set([int(ob[1]) for ob in list_ob]))
    list_index_ob.remove(ref_ob)
    list_ag_ob = list(rd.choice(list_index_ob, num_ag - 1, replace=False))
    list_ag_ob = list_ag_ob + [ref_ob]

    #   参照物サンプリング
    def creat_rel_samp(para):
        [mu_theta, ka_theta, mn, std] = para
        theta = rd.vonmises(mu_theta, ka_theta)
        r = abs(rd.normal(mn, std))
        X = [np.cos(theta), np.sin(theta), r, 1]
        return np.array(X).T

    ref_data = [creat_rel_samp(para[ref_cp])]

    #   非参照物のサンプリング
    def creat_unr_samp(num_unr):
        theta = rd.uniform(-pi, pi, num_unr)
        r = rd.uniform(.01, 5, num_unr)
        X = [np.cos(theta), np.sin(theta), r, [0 for i in range(num_unr)]]
        return np.array(X).T

    non_data = creat_unr_samp(num_ag - 1)

    data = np.zeros((num_ag, 6))
    data[:, :4] = list(non_data) + ref_data
    data[:, -2] = ref_cp
    data[:, -1] = list_ag_ob
    return data

#   発話によりサンプリングする
def sampling(sp, num_ag, num_ob):
    size = len(sp)
    Samp = [creat_sample(sp[i], num_ag, num_ob)
            for i in range(size)]
    return np.array(Samp)


