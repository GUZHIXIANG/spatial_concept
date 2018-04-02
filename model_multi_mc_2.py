# -*- coding: utf-8 -*-
# @Author: GU_ZHIXIANG
# @Date:   2018-01-10 16:46:37
# @Last Modified by:   GU_ZHIXIANG
# @Last Modified time: 2018-03-01 18:34:31


import numpy as np
import numpy.random as rd
from numpy import linalg
import scipy.stats as st
from scipy.special import i0
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import progress
# from math import *
import itertools
import pandas as pd
import sys


class ReSyMM:
    """This is a method for learning the relative concepts of space using position data (at referent-level) and sentences (at symbol-level). Simultaneously, this method can evaluate words relevant to the concepts being learned."""

    def __init__(self, nu0=0., kappa0=1., m0=1., sigma0=1., mu0=0., lamda0=1., a0=1., b0=1., alpha=.5, beta=.1, max_iter=1000):
        # hyperparameters of vonmises distribution
        self.nu0 = nu0
        self.kappa0 = kappa0
        self.m0 = m0
        self.sigma0 = sigma0
        # hyperparameters of normal distribution
        self.mu0 = mu0
        self.lamda0 = lamda0
        self.a0 = a0
        self.b0 = b0
        # hyperparameters of multinomial distribution
        self.alpha = alpha
        self.beta = beta
        # maximum number of iterations
        self.max_iter = max_iter

    ##############################
    '''Parameter initialization'''
    ##############################

    def _init_c(self):
        return np.zeros(self._N).astype(int) - 1

    def _init_z(self):
        return np.zeros(self._N).astype(int) - 1

    def _init_e(self):
        return [[] for i in range(self._N)]

    def _init_lamda(self):
        return rd.uniform(1, 2)

    def _init_mu(self):
        return rd.uniform(0, 1)

    def _init_nu(self):
        return rd.uniform(-np.pi, np.pi)

    def _init_kappa(self):
        return rd.uniform(1, 2)

    def _init_pi(self):
        return np.zeros(self._D) + 1. / self._D

    ########################
    '''Parameter sampling'''
    ########################

    def _sample_z(self):

        # Chinese Restaurant Precoss
        for i in range(self._N):

            old_r = self.c[i]
            if old_r != -1:
                if self.n[old_r] == 1:
                    del self.n[old_r]
                    del self.nu[old_r]
                    del self.kappa[old_r]
                    del self.pi[old_r]
                else:
                    self.n[old_r] -= 1
            like_ref, like_sym = [], []
            classes = {}
            _r = 0

            x = self._X[i]
            sent = self._Sent[i]
            like_dis = st.norm.pdf(
                x[:, 2], self.mu, np.sqrt(1. / self.lamda))
            prob_dis = like_dis / like_dis.sum()
            ang = np.arctan2(x[:, 1], x[:, 0])

            for r in self.n:
                if r == self.new_class:
                    nu = self._init_nu()
                    kappa = self._init_kappa()
                    pi = self._init_pi()
                    phi = self.alpha
                    new_nu = copy.copy(nu)
                    new_kappa = copy.copy(kappa)
                    new_pi = copy.copy(pi)
                else:
                    nu = self.nu[r]
                    kappa = self.kappa[r]
                    pi = self.pi[r]
                    phi = self.n[r]
                like_ref.append(st.vonmises.pdf(ang, kappa, nu) * phi)
                like_sym.append(pi[sent])
                classes[_r] = r
                _r += 1

            R = len(classes)
            K_n = len(x)
            D_n = len(sent)

            like_ref = np.array(like_ref) * prob_dis
            prob_ref = like_ref / like_ref.sum()
            like_sym = np.array(like_sym)
            prob_sym = like_sym / like_sym.sum()

            # the joint probability between concept,referent,symbol, P_r,k * P_r,d
            like_rkd = np.array([[[prob_ref[r, k] * prob_sym[r, d] for d in range(D_n)] for k in range(
                K_n)] for r in range(R)])

            prob_rkd = like_rkd / like_rkd.sum()
            prob_rkd = prob_rkd.reshape(R * K_n * D_n)
            C_rkd = rd.multinomial(1, prob_rkd).reshape((R, K_n, D_n))
            new_r = classes[np.where(C_rkd == 1)[0][0]]
            new_k = np.where(C_rkd == 1)[1]
            new_d = sent[np.where(C_rkd == 1)[2][0]]
            self.c[i] = new_r
            self.z[i] = new_k
            self.e[i] = new_d
            self.n[new_r] += 1

            if new_r == self.new_class:
                self.nu[self.new_class] = new_nu
                self.kappa[self.new_class] = new_kappa
                self.pi[self.new_class] = new_pi
                j = 0
                while True:
                    if not j in self.n:
                        self.n[j] = 0
                        self.new_class = j
                        break
                    j += 1

    def _calc_dis(self):
        dis_bar = 0
        for i in range(self._N):
            k = self.z[i]
            for k_ in k:
                dis_bar += self._X[i][k_, 2]
        return dis_bar / np.sum(list(self.n.values()))

    def _calc_ang(self):
        ang_bar = {}
        for i in range(self._N):
            r = self.c[i]
            k = self.z[i]
            if not r in ang_bar:
                ang_bar[r] = np.zeros(2)
            for k_ in k:
                ang_bar[r] += self._X[i][k_, :2]
        return ang_bar

    def _calc_word(self):
        word_bar = {}
        for i in range(self._N):
            r = self.c[i]
            sent = self._Sent[i]
            d = self.e[i]
            if not r in word_bar:
                word_bar[r] = np.zeros(self._D)
            word_bar[r][d] += 1
        return word_bar

    def _sample_mu(self):
        dis_bar = self._calc_dis()
        loc = (self._N / (self._N + self.lamda0)) * dis_bar + \
            (self.lamda0 / (self._N + self.lamda0)) * self.mu0
        scale = np.sqrt(1. / (self.lamda * (self._N + self.lamda0)))
        self.mu = rd.normal(loc, scale)

    def _sample_lamda(self):
        dis_bar = self._calc_dis()
        a = self.a0 + .5 * self._N
        b = self.b0 + (.5 * self._N * self.lamda0 /
                       (self._N + self.lamda0)) * (dis_bar - self.mu0)**2
        for i in range(self._N):
            k = self.z[i]
            b += np.sum(.5 * (self._X[i][k, 2] - dis_bar)**2)
        self.lamda = rd.gamma(a, 1. / b)

    def _sample_nu(self):
        ang_bar = self._calc_ang()
        nu0 = np.array([np.cos(self.nu0), np.sin(self.nu0)])
        for r in self.nu:
            tmp = self.kappa[r] * ang_bar[r] + nu0 * self.kappa0
            kappa = linalg.norm(tmp)
            nu = tmp / kappa
            nu = np.arctan2(nu[1], nu[0])
            self.nu[r] = rd.vonmises(nu, kappa)

    def _sample_kappa(self):
        ang_bar = self._calc_ang()
        for r in self.kappa:
            for loop in range(10):
                nu = [np.cos(self.nu[r]), np.sin(self.nu[r])]
                kappa_current = self.kappa[r]
                kappa_proposal = rd.lognormal(self.m0, self.sigma0)
                like_current = self._kappa_like(
                    kappa_current, nu, self.n[r], ang_bar[r])
                like_proposal = self._kappa_like(
                    kappa_proposal, nu, self.n[r], ang_bar[r])
                prior_current = st.lognorm(
                    s=self.sigma0, scale=np.exp(self.m0)).pdf(kappa_current)
                prior_proposal = st.lognorm(
                    s=self.sigma0, scale=np.exp(self.m0)).pdf(kappa_proposal)
                p_current = like_current * prior_current
                p_proposal = like_proposal * prior_proposal
                p_accept = p_proposal / p_current
                accept = np.random.rand() < p_accept
                if accept:
                    self.kappa[r] = copy.copy(kappa_proposal)

    def _kappa_like(self, kappa, nu, n, ang_bar):
        # the likelihood function of kappa
        return np.exp(n * np.log(1 / (2 * np.pi * i0(kappa))) + kappa * np.dot(nu, ang_bar))

    def _sample_pi(self):
        word_bar = self._calc_word()
        for r in self.pi:
            self.pi[r] = rd.dirichlet(word_bar[r] * self._isf + self.beta)

    def _select_word(self):
        word = {}
        for r in self.pi:
            word[r] = self._Dict[np.argmax(self.pi[r])]
        return word

    ##############
    '''Fragment'''
    ##############

    def _speech_digitization(self):
        '''
        output
            Dict: the dictionary of words
            Sent: the id set of each sentence
            isf: the inverse frequency (idf) of words in the whole data
        '''
        Dict = list(set(itertools.chain.from_iterable(self._Sp)))
        Sent = []
        isf = np.zeros(len(Dict))
        for i, sp in enumerate(self._Sp):
            tmp = []
            for l, word in enumerate(Dict):
                if word in sp:
                    tmp.append(l)
                    isf[l] += 1
            Sent.append(np.array(tmp))
        isf = np.log(len(Sent) / isf)
        return Dict, Sent, isf

    ########################
    '''Learning algorithm'''
    ########################

    def fit(self, X, Sp, path):
        self._X = X
        self._N = len(X)
        self._Sp = Sp
        self._Dict, self._Sent, self._isf = self._speech_digitization()
        self._D = len(self._Dict)
        self.new_class = 0
        self.n = {self.new_class: 0}
        self.c = self._init_c()
        self.z = self._init_z()
        self.e = self._init_e()
        self.pi = {}
        self.lamda = self._init_lamda()
        self.mu = self._init_mu()
        self.nu = {}
        self.kappa = {}
        self.construct = []
        self.remained_iter = copy.copy(self.max_iter)
        prog = progress.progressbar('Training', self.remained_iter - 1, '>')
        while True:
            self._sample_z()
            self._sample_mu()
            self._sample_lamda()
            self._sample_nu()
            self._sample_kappa()
            self._sample_pi()
            n = copy.copy(self.n)
            del n[self.new_class]
            self.construct.append(len(n))
            # if self.remained_iter % 25 == 0:
            #     self.plot_rf2cp(path, self.remained_iter)
            if self.remained_iter <= 0:
                break
            prog.progress(self.max_iter - self.remained_iter)
            self.remained_iter -= 1
        self.word_r = self._select_word()
        return self

    ############
    '''Output'''
    ############

    def parameters(self, path, loop):
        n = copy.copy(self.n)
        del n[self.new_class]
        para_recog = np.array(
            [list(self.nu.values()), list(self.kappa.values()), list(n.values())])
        para = pd.DataFrame(para_recog)
        # para.columns = list(self.n.keys())
        para.to_csv('%s/para_ver_master_%s.csv' % (path, loop))
        constt = pd.DataFrame(self.construct)
        constt .to_csv('%s/constt_ver_master_%s.csv' % (path, loop))
        return para_recog, self.mu, self.lamda

    def plot_rf2cp(self, path, loop):
        plt.figure(figsize=(8, 4))
        ymajorFormatter = FormatStrFormatter('%1.1f')
        # ax = plt.subplot2grid((10, 1), (0, 0), rowspan=5)
        ax = plt.subplot(111)
        ax.yaxis.set_major_formatter(ymajorFormatter)
        plt.xlim([-180, 180])
        plt.ylim([0, 2])
        plt.xlabel('Angle($\circ$)', fontsize=9)
        plt.ylabel('the distribution of angle', fontsize=8)
        plt.sca(ax)
        x_ang = np.linspace(-np.pi, np.pi, 1000)
        for r in self.nu:
            y_ang = st.vonmises.pdf(x_ang, self.kappa[r], self.nu[r])
            ax.plot(x_ang * 180 / np.pi, y_ang, '--',
                    linewidth=2.3, label="$r$=%d" % (r))
        plt.legend(bbox_to_anchor=(.8, .95), shadow=True, fontsize=8)
        plt.savefig("%s/dis_ver_master_%d.png" % (path, loop), dpi=600)
        plt.close()

    def plot_cp2sb(self, path, loop):
        sns.set_style("darkgrid")
        sns.palplot(sns.color_palette("hls", 8))
        font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
        font_prop = FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = font_prop.get_name()

        labels = ["r_{}".format(i) for i in self.n]
        data = np.r_[np.array(list(self.pi.values()))]
        for label, y in zip(labels, data):

            plt.figure(figsize=(10, 5))
            ax = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
            ymajorFormatter = FormatStrFormatter('%1.1f')
            ax.yaxis.set_major_formatter(ymajorFormatter)
            x = np.arange(10) - 1
            y_max_indices = np.argsort(y)[::-1][:10]
            word_list = [self._Dict[i] for i in y_max_indices]
            y = np.sort(y)[::-1][:10]
            saved_data = pd.DataFrame(y)
            saved_data.to_csv('%s/import_ver_master_%s.csv' %
                              (path, label), encoding='utf-8')
            plt.xticks(x, word_list, size='small', rotation=30)
            plt.ylabel("Import")
            plt.bar(x, y, width=.35, align='center', color='r', alpha=0.8)
            plt.savefig("%s/import_ver_master_%d_%s.png" %
                        (path, loop, label), dpi=600)
            plt.close()

    def predict_sentence(self, path, loop):
        imp = np.zeros((self._N, len(self.pi)))
        recog_pi = np.zeros((len(self.pi), self._D))
        i = 0
        for r in self.pi:
            recog_pi[i] = self.pi[r]
            i += 1
        for i in range(self._N):
            sent = self._Sent[i]
            imp[i] = recog_pi[:, sent].sum(axis=1)
        imp = pd.DataFrame(imp)
        imp.to_csv('%s/pre_ver_master_%s.csv' % (path, loop))
