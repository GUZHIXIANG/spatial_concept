# -*- coding: utf-8 -*-
# @Author: GU_ZHIXIANG
# @Date:   2018-04-02 13:23:12
# @Last Modified by:   GU_ZHIXIANG
# @Last Modified time: 2018-04-02 15:50:55

import numpy as np
import numpy.random as rd
from numpy import linalg
import scipy.stats as st
from scipy.special import i0
import copy
import seaborn as sns
import itertools
import sys
from utility import progressbar


class ReSyMM:
    """This is a method for learning the relative concepts of space using position data (at referent-level) and sentences (at symbol-level). Simultaneously, this method can evaluate words relevant to the concepts being learned."""

    def __init__(self, R=None, nu0=0., kappa0=1., m0=1., sigma0=1., mu0=0., lamda0=1., a0=1., b0=1., alpha=1., beta=.1, max_iter=1000):
        self.R = R
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
        return np.zeros(self.R) + 1. / self.R

    def _init_phi(self):
        return np.zeros(self._D) + 1. / self._D

    ########################
    '''Parameter sampling'''
    ########################

    def _sample_cze(self):
        self.n = {r: 0 for r in range(self.R)}
        for i in range(self._N):
            like_ref, like_sym = [], []
            x = self._X[i]
            sent = self._Sent[i]
            like_dis = st.norm.pdf(
                x[:, 2], self.mu, np.sqrt(1. / self.lamda))
            prob_dis = like_dis / like_dis.sum()
            ang = np.arctan2(x[:, 1], x[:, 0])

            for r in self.n:
                nu = self.nu[r]
                kappa = self.kappa[r]
                phi = self.phi[r]
                pi = self.pi[r]
                like_ref.append(st.vonmises.pdf(ang, kappa, nu) * pi)
                like_sym.append(phi[sent])

            R = self.R
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
            self.c[i] = np.where(C_rkd == 1)[0][0]
            self.z[i] = np.where(C_rkd == 1)[1][0]
            self.e[i] = sent[np.where(C_rkd == 1)[2][0]]
            self.n[self.c[i]] += 1

    def _sample_cze_nonpara(self):

        # Chinese Restaurant Precoss
        for i in range(self._N):
            old_r = self.c[i]
            if old_r != -1:
                if self.n[old_r] == 1:
                    del self.n[old_r]
                    del self.nu[old_r]
                    del self.kappa[old_r]
                    del self.phi[old_r]
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
                    phi = self._init_phi()
                    pi = self.alpha
                    new_nu = copy.copy(nu)
                    new_kappa = copy.copy(kappa)
                    new_phi = copy.copy(phi)
                else:
                    nu = self.nu[r]
                    kappa = self.kappa[r]
                    phi = self.phi[r]
                    pi = self.n[r]
                like_ref.append(st.vonmises.pdf(ang, kappa, nu) * pi)
                like_sym.append(phi[sent])
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
            new_k = np.where(C_rkd == 1)[1][0]
            new_d = sent[np.where(C_rkd == 1)[2][0]]
            self.c[i] = new_r
            self.z[i] = new_k
            self.e[i] = new_d
            self.n[new_r] += 1

            if new_r == self.new_class:
                self.nu[self.new_class] = new_nu
                self.kappa[self.new_class] = new_kappa
                self.phi[self.new_class] = new_phi
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
            dis_bar += self._X[i][k, 2]
        return dis_bar / np.sum(list(self.n.values()))

    def _calc_ang(self):
        ang_bar = {}
        if self.R is not None:
            ang_bar = {r: np.zeros(2) for r in range(self.R)}
        for i in range(self._N):
            r = self.c[i]
            k = self.z[i]
            if not r in ang_bar:
                ang_bar[r] = np.zeros(2)
            ang_bar[r] += self._X[i][k, :2]
        return ang_bar

    def _calc_word(self):
        word_bar = {}
        if self.R is not None:
            word_bar = {r: np.zeros(self._D) for r in range(self.R)}
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
        self.pi = rd.dirichlet(np.array(list(self.n.values())) + self.beta)

    def _sample_phi(self):
        word_bar = self._calc_word()
        for r in self.phi:
            self.phi[r] = rd.dirichlet(word_bar[r] * self._isf + self.beta)

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

    def _select_word(self):
        word = {}
        for r in self.phi:
            word[r] = self._Dict[np.argmax(self.phi[r])]
        return word

    ########################
    '''Learning algorithm'''
    ########################

    def fit(self, X, Sp, path):
        self._X = X
        self._N = len(X)
        self._Sp = Sp
        self._Dict, self._Sent, self._isf = self._speech_digitization()
        self._D = len(self._Dict)
        self.c = self._init_c()
        self.z = self._init_z()
        self.e = self._init_e()
        self.phi = {}
        self.lamda = self._init_lamda()
        self.mu = self._init_mu()
        self.nu = {}
        self.kappa = {}
        self.new_class = 0
        self.n = {self.new_class: 0}
        cze_sampler = self._sample_cze_nonpara
        if self.R is not None:
            for r in range(self.R):
                self.phi[r] = self._init_phi()
                self.nu[r] = self._init_nu()
                self.kappa[r] = self._init_kappa()
            self.pi = self._init_pi()
            cze_sampler = self._sample_cze
        self.construct = []
        self.remained_iter = copy.copy(self.max_iter)
        prog = progressbar('Training', self.remained_iter - 1, '>')
        while True:
            cze_sampler()
            self._sample_mu()
            self._sample_lamda()
            self._sample_nu()
            self._sample_kappa()
            self._sample_phi()
            if self.R is not None:
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


class ReSyMMwithObject(ReSyMM):

    ##############################
    '''Parameter initialization'''
    ##############################

    def _init_psi(self):
        return np.zeros((self._O, self._D)) + + 1. / self._D

    ########################
    '''Parameter sampling'''
    ########################

    def _sample_cze(self):
        # Chinese Restaurant Precoss
        for i in range(self._N):
            old_r = self.c[i]
            if old_r != -1:
                if self.n[old_r] == 1:
                    del self.n[old_r]
                    del self.nu[old_r]
                    del self.kappa[old_r]
                    del self.phi[old_r]
                else:
                    self.n[old_r] -= 1
            like_ref, like_sym_s = [], []
            classes = {}
            _r = 0

            x = self._X[i]
            sent = self._Sent[i]
            index = self._Index[i].astype(int)
            like_dis = st.norm.pdf(
                x[:, 2], self.mu, np.sqrt(1. / self.lamda))
            prob_dis = like_dis / like_dis.sum()
            ang = np.arctan2(x[:, 1], x[:, 0])
            like_sym_o = np.mat(self.psi[index][:, sent])

            for r in self.n:
                if r == self.new_class:
                    nu = self._init_nu()
                    kappa = self._init_kappa()
                    phi = self._init_phi()  # concept word prob
                    pi = self.alpha
                    new_nu = copy.copy(nu)
                    new_kappa = copy.copy(kappa)
                    new_phi = copy.copy(phi)
                else:
                    nu = self.nu[r]
                    phi = self.phi[r]
                    kappa = self.kappa[r]
                    pi = self.n[r]
                like_ref.append(st.vonmises.pdf(ang, kappa, nu) * pi)
                like_sym_s.append(phi[sent])
                classes[_r] = r
                _r += 1

            R = len(classes)
            K_n = len(x)
            D_n = len(sent)

            like_ref = np.array(like_ref) * prob_dis
            prob_ref = like_ref / like_ref.sum()
            like_sym_s = np.array(like_sym_s)
            prob_sym_s = like_sym_s / like_sym_s.sum()
            like_sym_o = np.array(like_sym_o)
            prob_sym_o = like_sym_o / like_sym_o.sum()

            # the joint probability between concept,referent,symbol, P_r,k * P_r,d * P_k,_d
            like_rkd_d = np.array([[[[prob_ref[r, k] * prob_sym_s[r, d] * prob_sym_o[k, _d] if _d != d else 0 for _d in range(D_n)] for d in range(D_n)] for k in range(
                K_n)] for r in range(R)])

            prob_rkd_d = like_rkd_d / like_rkd_d.sum()
            prob_rkd_d = prob_rkd_d.reshape(R * K_n * D_n * D_n)
            C_rkd_d = rd.multinomial(1, prob_rkd_d).reshape((R, K_n, D_n, D_n))
            # concept
            new_r = classes[np.where(C_rkd_d == 1)[0][0]]
            # reference
            new_k = np.where(C_rkd_d == 1)[1][0]
            # concept word
            new_d_s = sent[np.where(C_rkd_d == 1)[2][0]]
            # reference word
            new_d_o = sent[np.where(C_rkd_d == 1)[3][0]]

            self.c[i] = new_r
            self.z[i] = new_k
            self.e[i] = [new_d_s, new_d_o]
            self.n[new_r] += 1

            if new_r == self.new_class:
                self.nu[self.new_class] = new_nu
                self.kappa[self.new_class] = new_kappa
                self.phi[self.new_class] = new_phi
                j = 0
                while True:
                    if not j in self.n:
                        self.n[j] = 0
                        self.new_class = j
                        break
                    j += 1

    def _calc_word(self):
        word_bar_s = {}
        word_bar_o = np.zeros((self._O, self._D))
        for i in range(self._N):
            r = self.c[i]
            e = self.e[i]
            if not r in word_bar_s:
                word_bar_s[r] = np.zeros(self._D)
            word_bar_s[r][e[0]] += 1
            k = self.z[i]
            index = self._Index[i].astype(int)
            word_bar_o[index[k], e[1]] += 1
        return word_bar_s, word_bar_o

    def _sample_phi(self):
        word_bar_s, word_bar_o = self._calc_word()
        for r in self.phi:
            self.phi[r] = rd.dirichlet(word_bar_s[r] * self._isf + self.beta)
        self.psi = np.array([rd.dirichlet(word_bar_o[id_o] * self._isf + self.beta)
                             for id_o in range(self._O)])

    ##############
    '''Fragment'''
    ##############

    def _detect_index(self):
        _index = []
        set_index = []
        for x in self._X:
            _index.extend(x[:, -1])
            set_index.append(x[:, -1])
        return len(list(set(_index))), set_index

    def _select_word(self):
        word_s = {}
        word_o = {}
        for r in self.phi:
            word_s[r] = self._Dict[np.argmax(self.phi[r])]
        for id_o in range(self._O):
            word_o[id_o] = self._Dict[np.argmax(self.psi[id_o])]
        return word_s, word_o

    ########################
    '''Learning algorithm'''
    ########################

    def fit(self, X, Sp, path):
        self._X = X
        self._N = len(X)
        self._Sp = Sp
        self._Dict, self._Sent, self._isf = self._speech_digitization()
        self._D = len(self._Dict)
        self._O, self._Index = self._detect_index()
        self.new_class = 0
        self.n = {self.new_class: 0}
        self.c = self._init_c()
        self.z = self._init_z()
        self.e = self._init_e()
        self.phi = {}
        self.psi = self._init_psi()
        self.lamda = self._init_lamda()
        self.mu = self._init_mu()
        self.nu = {}
        self.kappa = {}
        self.construct = []
        self.remained_iter = copy.copy(self.max_iter)
        prog = progressbar('Training', self.remained_iter - 1, '>')
        while True:
            self._sample_cze()
            self._sample_mu()
            self._sample_lamda()
            self._sample_nu()
            self._sample_kappa()
            self._sample_phi()
            n = copy.copy(self.n)
            del n[self.new_class]
            self.construct.append(len(n))
            if self.remained_iter <= 0:
                break
            prog.progress(self.max_iter - self.remained_iter)
            self.remained_iter -= 1
        self.word_s, self.word_o = self._select_word()
        return self
