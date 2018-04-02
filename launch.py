# -*- coding: utf-8 -*-
# @Author: GU_ZHIXIANG
# @Date:   2018-04-02 13:24:47
# @Last Modified by:   GU_ZHIXIANG
# @Last Modified time: 2018-04-02 16:25:00

import os
import sys
import numpy.random as rd
import simulator
from model import ReSyMM, ReSyMMwithObject
from utility import parameters, plot_rf2cp, plot_cp2sb, plot_cp2sb_with_object, progressbar

if __name__ == '__main__':

    # rd.seed(0)

    num_ob = 16  # 物体概念数（既知）
    num_ag = 5  # 配置物体数

    '''########################################'''
    '''<<<<<<<<<Data Generating Modul>>>>>>>>>>'''
    '''########################################'''

    with open('speech_data/true_%dx4x4_en.txt' % num_ob, 'r', encoding='utf-8') as file:
        sp_true = []
        for line in file:
            sentence = line.split()
            sp_true.append(sentence)

    with open('speech_data/recog_%dx4x4_en.txt' % num_ob, 'r', encoding='utf-8') as file:
        sp_recog = []
        for line in file:
            sentence = line.split()
            sp_recog.append(sentence)

    # makedirs
    path = 'test_%d_%d' % (num_ob, num_ag)
    if not os.path.exists(path):
        os.makedirs(path)

    # stochastic data
    data = simulator.sampling(sp_true, num_ag, num_ob)
    # print (data)

    '''#################################'''
    '''<<<<<<<<<Learning Modul>>>>>>>>>>'''
    '''#################################'''

    # learning without object index
    # model = ReSyMM(max_iter=1000)
    # model.fit(data, sp_true, path)
    # print(parameters(model, path))  # output parameters estimated by learning
    # plot_rf2cp(model, path)  # plot distributions of relative positions
    # plot_cp2sb(model, path)  # plot distributions of words of concepts

    # learning with object index
    model = ReSyMMwithObject(max_iter=500)
    model.fit(data, sp_recog, path)
    print(parameters(model, path))
    plot_rf2cp(model, path)
    plot_cp2sb_with_object(model, path)
