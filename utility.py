# -*- coding: utf-8 -*-
# @Author: GU_ZHIXIANG
# @Date:   2018-04-02 13:24:50
# @Last Modified by:   GU_ZHIXIANG
# @Last Modified time: 2018-04-02 16:05:02

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd


def parameters(model, path):
    n = copy.copy(model.n)
    para_recog = np.array(
        [list(model.nu.values()), list(model.kappa.values()), list(n.values())])
    para = pd.DataFrame(para_recog)
    # para.columns = list(model.n.keys())
    para.to_csv('%s/parameters.csv' % (path))
    constt = pd.DataFrame(model.construct)
    constt.to_csv('%s/constt.csv' % (path))
    return para_recog, model.mu, model.lamda


def plot_rf2cp(model, path):
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
    for r in model.nu:
        if model.n[r] < 5:
            continue
        y_ang = st.vonmises.pdf(x_ang, model.kappa[r], model.nu[r])
        ax.plot(x_ang * 180 / np.pi, y_ang, '--',
                linewidth=2.3, label="%s" % (model.word_s[r]))
    plt.legend(bbox_to_anchor=(.8, .95), shadow=True, fontsize=8)
    plt.savefig("%s/distribution.png" % (path), dpi=600)
    plt.close()


def plot_cp2sb(model, path):
    sns.set_style("darkgrid")
    sns.palplot(sns.color_palette("hls", 8))
    # font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
    # font_prop = FontProperties(fname=font_path)
    # matplotlib.rcParams['font.family'] = font_prop.get_name()

    labels = ["r_{}".format(i) for i in model.n]
    data = np.r_[np.array(list(model.phi.values()))]
    for label, y in zip(labels, data):

        plt.figure(figsize=(10, 5))
        ax = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
        ymajorFormatter = FormatStrFormatter('%1.1f')
        ax.yaxis.set_major_formatter(ymajorFormatter)
        x = np.arange(10) - 1
        y_max_indices = np.argsort(y)[::-1][:10]
        word_list = [model._Dict[i] for i in y_max_indices]
        y = np.sort(y)[::-1][:10]
        saved_data = pd.DataFrame(y)
        saved_data.to_csv('%s/import_%s.csv' %
                          (path, label), encoding='utf-8')
        plt.xticks(x, word_list, size='small', rotation=30)
        plt.ylabel("Import")
        plt.bar(x, y, width=.35, align='center', color='r', alpha=0.8)
        plt.savefig("%s/import_%s.png" %
                    (path, label), dpi=600)
        plt.close()


def plot_cp2sb_with_object(model, path):
    sns.set_style("darkgrid")
    sns.palplot(sns.color_palette("hls", 8))
    # font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
    # font_prop = FontProperties(fname=font_path)
    # matplotlib.rcParams['font.family'] = font_prop.get_name()
    n = copy.copy(model.n)
    del n[model.new_class]
    labels = ["spatial_{}".format(
        i) for i in n] + ["object_{}".format(id_o) for id_o in range(model._O)]
    data = np.r_[np.array(list(model.phi.values())), model.psi]
    for label, y in zip(labels, data):
        plt.figure(figsize=(10, 5))
        ax = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
        ymajorFormatter = FormatStrFormatter('%1.1f')
        ax.yaxis.set_major_formatter(ymajorFormatter)
        x = np.arange(10) - 1
        y_max_indices = np.argsort(y)[::-1][:10]
        word_list = [model._Dict[i] for i in y_max_indices]
        y = np.sort(y)[::-1][:10]
        saved_data = pd.DataFrame(y)
        saved_data.to_csv('%s/import_%s.csv' %
                          (path, label), encoding='utf-8')
        plt.xticks(x, word_list, size='small', rotation=30)
        plt.ylabel("Import")
        plt.bar(x, y, width=.35, align='center', color='r', alpha=0.8)
        plt.savefig("%s/import_%s.png" %
                    (path, label), dpi=600)
        plt.close()


def predict_sentence(model, path):
    imp = np.zeros((model._N, len(model.pi)))
    recog_pi = np.zeros((len(model.pi), model._D))
    i = 0
    for r in model.pi:
        recog_pi[i] = model.pi[r]
        i += 1
    for i in range(model._N):
        sent = model._Sent[i]
        imp[i] = recog_pi[:, sent].sum(axis=1)
    imp = pd.DataFrame(imp)
    imp.to_csv('%s/predict.csv' % (path))


class progressbar:
    def __init__(self, topic, finalcount, progresschar=None):
        import sys
        self.finalcount = finalcount
        self.blockcount = 0
        #
        # See if caller passed me a character to use on the
        # progress bar (like "*").  If not use the block
        # character that makes it look like a real progress
        # bar.
        #
        if not progresschar:
            self.block = chr(178)
        else:
            self.block = progresschar
        #
        # Get pointer to sys.stdout so I can use the write/flush
        # methods to display the progress bar.
        #
        self.f = sys.stdout
        #
        # If the final count is zero, don't start the progress gauge
        #
        if not self.finalcount:
            return
        self.f.write(
            '\n-------------------- %s --------------------\n' % topic)
        return

    def progress(self, count):
        #
        # Make sure I don't try to go off the end (e.g. >100%)
        #
        count = min(count, self.finalcount)
        #
        # If finalcount is zero, I'm done
        #
        if self.finalcount:
            percentcomplete = int(round(100 * count / self.finalcount))
            if percentcomplete < 1:
                percentcomplete = 1
        else:
            percentcomplete = 100

        # print "percentcomplete=",percentcomplete
        blockcount = int(percentcomplete / 2)
        # print "blockcount=",blockcount
        if blockcount > self.blockcount:
            for i in range(self.blockcount, blockcount):
                self.f.write(self.block)
                self.f.flush()

        if percentcomplete == 100:
            self.f.write("\n")
        self.blockcount = blockcount
        return
