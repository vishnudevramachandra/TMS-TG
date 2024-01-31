import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import seaborn as sns
import copy
import pickle

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import (adjust_lim, plot_populationAvgFR, normalize_psfr, plot_delay, fb)


def plot(tms, activeNeus, xlim=None):
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()
    delays = dict()
    fig, ax = plt.subplots(2, 1)

    postCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'Skin-Injection': 'Post'}}
    tms.analysis_params = postCond
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
    epochIndices = selectBlocksinfo.index.unique()

    if not any(selectBlocksinfoIdx):
        print(f'Cannot plot Skin-Injection as the associated data is missing in this group of animals')
        return

    # statistics
    animalNumsEpochNumsAndActiveNeuronNums_perCol.append((np.unique([item[0] for item in epochIndices]).size,
                                                          len(epochIndices),
                                                          [activeNeus[item].sum() for item in epochIndices]))

    # # lineplot
    # ps_FR, ps_T = tms.compute_firingrate(
    #     *tms.analysis_params['peristimParams']['smoothingParams'].values(),
    #     *tms.analysis_params['peristimParams']['timeWin'],
    #     tms.analysis_params['peristimParams']['trigger'])
    # ps_T_corrected = tms.analysis_params['peristimParams']['timeWin'][0] + ps_T

    # delay
    meanPSFR, t, meanBaselineFR, _ = tms.avg_FR_per_neuron(squeezeDim=False)
    _, zeroMTIdx = fb(selectBlocksinfo, zeroMTCond)
    injWithPost = selectBlocksinfo['Skin-Injection'].str.contains('Post')
    postT = selectBlocksinfo['Skin-Injection'].str.extract('(\d+)').astype('float64')[0]
    delay = list()
    for ti in [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)]:
        blockIdx = (ti[0] <= postT) & (postT < ti[1]) & ~zeroMTIdx
        boolIndex = [activeNeu[item] if activeNeu[item].size > 1 else activeNeu[item][np.newaxis]
                     for item in selectBlocksinfo.loc[blockIdx].index]
        selectMeanPSFR = np.concatenate([meanPSFR[n][:, :, item][0, :, :]
                                         for n, item in zip(np.flatnonzero(blockIdx), boolIndex)],
                                        axis=1)
        d = tms.late_comp.compute_delay(selectMeanPSFR,
                                        t,
                                        selectMeanPSFR[t < 0, :].max(axis=0, keepdims=True),
                                        tms.analysis_params['peristimParams']['smoothingParams'][
                                            'width'] + 0.25)
        delay.append(d)
    fig, ax = plt.subplots(1, 1)
    sns.swarmplot(data=delay, color='k', size=3)
    sns.violinplot(data=delay, inner=None, ax=ax)
    ax.set_xticks([0, 1, 2, 3, 4], labels=['0-10 mins', '10-20 mins', '20-30 mins', '30-40 mins', '40-108 mins'])
    ax.set_ylabel('Delay (ms)', fontsize=12)
    ax.set_title('Skin-Injection', fontsize=12)
    plt.savefig('figure3a.pdf', dpi='figure', format='pdf')
    plt.show()



if __name__ == '__main__':
    epochs = [('20190714', 'MC', 'L5', 'none', 'none', '1400'),
              ('20191119', 'MC', 'L5', 'none', 'none', '1330'),
              ('20191201', 'MC', 'L5', 'none', 'none', '1300'),
              ('20191201', 'MC', 'L5', 'none', 'none', '1301')]
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    # activeNeu = tms.stats_is_signf_active()
    activeNeu = pd.read_pickle("./activeNeu_Cortex")
    plot(tms, activeNeu, xlim=[-20, 60])