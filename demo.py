# -*- coding: utf-8 -*-
# @Author: GU_ZHIXIANG
# @Date:   2018-01-06 21:31:54
# @Last Modified by:   gzxsp
# @Last Modified time: 2018-03-31 17:34:40

import os
import sys
import numpy.random as rd
import simulator
import model_tfif_np_4
import model_multi_mc_2

if __name__ == '__main__':

    # rd.seed(0)

    num_ob = 16  # 物体概念数（既知）
    num_ag = 5  # 配置物体数

    '''########################################'''
    '''<<<<<<<<<Data Generating Modul>>>>>>>>>>'''
    '''########################################'''

    with open('speech_data/true_%dx4x4.txt' % num_ob, 'r', encoding='utf-8') as file:
        sp_true = []
        for line in file:
            sentence = line.split()
            sp_true.append(sentence)

    with open('speech_data/recog_%dx4x4.txt' % num_ob, 'r', encoding='utf-8') as file:
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

    # estimate object names & spatial concepts simultaneously
    model = model_tfif_np_4.ReSyMM(alpha=1., m0=1., sigma0=1, max_iter=1000)
    model.fit(data, sp_true, path)
    print(model.parameters(path))  # output parameters estimated by learning
    model.plot_rf2cp(path)  # plot distributions of relative positions
    # model.plot_cp2sb(path)  # plot distributions of words of concepts
    model.predict_sentence(path)  # predict aim of sentences

    # estimate spatial concepts only
    # model = model_multi_mc_2.ReSyMM(
    #     alpha=.5, m0=1., sigma0=1., max_iter=2000)
    # model.fit(data, sp_true, path)
    # print(model.parameters(path, loop))
    # model.plot_rf2cp(path, loop)
    # model.plot_cp2sb(path, loop)
    # model.predict_sentence(path, loop)
