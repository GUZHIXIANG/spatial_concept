# -*- coding: utf-8 -*-
# @Author: GU_ZHIXIANG
# @Date:   2018-01-06 21:31:54
# @Last Modified by:   GU_ZHIXIANG
# @Last Modified time: 2018-04-02 13:16:56
import os
import sys
import numpy as np
import numpy.random as rd
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import simulator
# import simulator
# import model_mc
# import model_multi_mc
# import model_lda
# import model_np_os
# import model_tfif_np_4
import model_tfif_np_3
# import model_multi_mc_2_9
# import model_multi_mc_2_9_wod
# import model_multi_mc_2_9_woi
# import model_multi_mc_2_9_wodi
# import model_multi_mc_2_9_0_1
# import model_multi_mc_3_2
# import model_multi_mc_3_1
# import signifer
from math import *
import copy


def plot_point(data, path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    # plt.style.use('ggplot')
    ax.set_xticks([])
    ax.set_yticks([])

    color = ['C0', 'C3']
    for dat in data:
        ang = dat[:, :2]
        dis = dat[:, 2]
        index = dat[:, -2].astype(int)

        coor = (ang.T * dis).T
        for k in range(len(coor)):
            plt.plot(coor[k, 0], coor[k, 1], '%s.' % color[index[k]])
    # plt.show()
    plt.savefig("%s/distribution_point.png" % path, dpi=600)
    plt.close()


if __name__ == '__main__':
    # rd.seed(0)

    # True speech
    num_ob = 16

    with open('speech_data/true_%dx4x4.txt' % num_ob, 'r') as file:
        sp_true = []
        for line in file:
            sentence = line.split()
            sp_true.append(sentence)

    with open('speech_data/recog_%dx4x4.txt' % num_ob, 'r') as file:
        sp_recog = []
        for line in file:
            sentence = line.split()
            sp_recog.append(sentence)

    for loop in range(1, 2):
        path = 'test_4/6/{}'.format(loop)
        if not os.path.exists(path):
            os.makedirs(path)
        # Stochastic data
        # data = simulator.sampling_with_speech(sp_true, 3)
        data = simulator.sampling(sp_true, 5, num_ob)
        # plot_point(data, path)\
        # print (data)

        '''#########################################'''
        '''<<<<<<<<<Concept Learning Modul>>>>>>>>>>'''
        '''#########################################'''

        # # single
        # model = model_mc.BeSRM(max_iter=500)
        # model.fit(data)
        # para = model.parameters()
        # print ("BeSRM", para)

        # # multi
        # model = model_multi_mc.ReSyMM(max_iter=500)
        # model.fit(data)
        # para_cp, para_ob = model.parameters(path)
        # # print "ReSyMM"
        # para = copy.copy(para_cp)
        # para[:, 0] = para[:, 0] * 180 / np.pi
        # print(para)
        # model.plot_rf2cp(path)

        # # Estimation
        # process = signifer.WIA(para_cp, para_ob)
        # process.estimate(sp, data)
        # process.save_result(path)
        # process.plot_cp2sb(path)

        # multi_object&spatial
        model = model_tfif_np_4.ReSyMM(alpha=1.)
        model.fit(data, sp_true, path)
        para = model.parameters(path, loop)
        # print "ReSyMM"
        print(para)
        model.plot_rf2cp(path, loop)
        model.plot_cp2sb(path, loop)
        # print(model.predict())
        model.predict_sentence(path, loop)

        # # multi_spatial
        # model = model_multi_mc_2_9_0_1.ReSyMM(
        #     alpha=.5, m0=1, sigma0=1., max_iter=2000)
        # model.fit(data, sp_true, path)
        # para = model.parameters(path, loop)
        # # print "ReSyMM"
        # print(para)
        # model.plot_rf2cp(path, loop)
        # model.plot_cp2sb(path, loop)
        # print(model.predict())
        # model.predict_sentence(path, loop)

        # # multi_2_7
        # model = model_multi_mc_2_9.ReSyMM(alpha=0.1, max_iter=2000)
        # model.fit(data, sp_true, path)
        # para = model.parameters(path, loop)
        # # print "ReSyMM"
        # print(para)
        # model.plot_rf2cp(path, loop)
        # model.plot_cp2sb(path, loop)
        # print(model.predict())

        # # model = model_multi_mc_2_9_wod.ReSyMM(alpha=0.1, max_iter=2000)
        # # model.fit(data, sp_true, path)
        # # para = model.parameters(path, loop)
        # # # print "ReSyMM"
        # # print(para)
        # # model.plot_rf2cp(path, loop)
        # # model.plot_cp2sb(path, loop)
        # # print(model.predict())

        # model = model_multi_mc_2_9_woi.ReSyMM(alpha=0.1, max_iter=2000)
        # model.fit(data, sp_true, path)
        # para = model.parameters(path, loop)
        # # print "ReSyMM"
        # print(para)
        # model.plot_rf2cp(path, loop)
        # model.plot_cp2sb(path, loop)
        # print(model.predict())
        # model.predict_sentence(path, loop)

        # model = model_multi_mc_2_9_wodi.ReSyMM(alpha=0.1, max_iter=2000)
        # model.fit(data, sp_true, path)
        # para = model.parameters(path, loop)
        # # print "ReSyMM"
        # print(para)
        # model.plot_rf2cp(path, loop)
        # model.plot_cp2sb(path, loop)
        # print(model.predict())

        # # Fixed lda
        # model = model_lda.ReSyMM(max_iter=200)
        # model.fit(data, sp_true)
        # # model.parameters(path)
        # model.plot_rf2cp(path)
        # model.plot_cp2sb(path)
