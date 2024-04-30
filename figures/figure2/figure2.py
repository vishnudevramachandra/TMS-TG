import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import copy
import pickle
import seaborn as sns

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import (adjust_lim, plot_populationAvgFR, normalize_psfr, ascertain_colName_from_colParams,
                                 pick_delay)


def plot(tms, activeNeus, colParam, axes, xlim=None, frKind=None, frYlim=None, delayYlim=None):
    traceConds = ({'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '<1'}},
                  {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}},
                  {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '>1'}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2')
    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()

    colName = colParam['selectionParams']['RecArea ']
    axes[0].set_title(colName, fontsize=8)
    tms.analysis_params = copy.deepcopy(colParam)
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
    epochIndices = selectBlocksinfo.index.unique()

    if not any(selectBlocksinfoIdx):
        print(f'Cannot plot for {colParam} as the associated data is missing in this group of animals')
        return frYlim, delayYlim

    ps_FR, ps_T = tms.compute_firingrate(
        *tms.analysis_params['peristimParams']['smoothingParams'].values(),
        *tms.analysis_params['peristimParams']['timeWin'],
        tms.analysis_params['peristimParams']['trigger'])
    ps_T_corrected = tms.analysis_params['peristimParams']['timeWin'][0] + ps_T

    # statistics
    animalNumsEpochNumsAndActiveNeuronNums_perCol.append((np.unique([item[0] for item in epochIndices]).size,
                                                          len(epochIndices),
                                                          [activeNeus[item].sum() for item in epochIndices]))

    # --------------------------------------------------------------------------------------------------------------
    # plot population activity
    match frKind:
        case 'Normalized':
            plot_populationAvgFR(normalize_psfr(ps_T_corrected, ps_FR, selectBlocksinfo), ps_T_corrected,
                                 selectBlocksinfo, zeroMTCond, activeNeus, traceConds, axes[0], colorsPlt,
                                 [item['selectionParams']['MT'] for item in traceConds])
        case _:
            plot_populationAvgFR(ps_FR, ps_T_corrected,
                                 selectBlocksinfo, zeroMTCond, activeNeus, traceConds, axes[0], colorsPlt,
                                 [item['selectionParams']['MT'] for item in traceConds])

    # --------------------------------------------------------------------------------------------------------------
    # plot delay
    sns.swarmplot(data=pick_delay(tms, selectBlocksinfo, zeroMTCond, traceConds, activeNeus),
                  color='k', size=3, ax=axes[1])
    sns.violinplot(data=list(delays[ascertain_colName_from_colParams(colParam)].values()),
                   inner=None, ax=axes[1])
    axes[1].set_xticks(range(len(traceConds)), labels=[item['selectionParams']['MT'] for item in traceConds])

    # --------------------------------------------------------------------------------------------------------------
    if xlim is not None:
        adjust_lim(axes, xlim, 'xlim')

    return ax[0].get_ylim(), ax[1].get_ylim()




    plt.savefig('figure2.pdf', dpi='figure', format='pdf')
    plt.show()
    with open('delays.pickle', 'wb') as f:
        pickle.dump(delays, f)
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums_perCol)


if __name__ == '__main__':
    # nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    epochs = (
        {'selectionParams': {'Epoch': {'Region': 'thal', }, 'RecArea ': 'VPM'}},
        {'selectionParams': {'Epoch': {'Region': 'thal', }, 'RecArea ': 'VPL'}},
        {'selectionParams': {'Epoch': {'Region': 'thal', }, 'RecArea ': 'VM'}},
        {'selectionParams': {'Epoch': {'Region': 'thal', }, 'RecArea ': 'PO'}},
    )

    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    # activeNeu = tms.stats_is_signf_active()

    dYlim = [np.inf, 0]
    fig, ax = plt.subplots(2, len(epochs))
    for colIdx in range(len(epochs)):
        frYlim, delayYlim = plot(
            tms, pd.read_pickle("./activeNeu_thal"), epochs[colIdx], ax[:, colIdx], xlim=[-20, 60], frKind='Average')
        if colIdx == 0:
            ax[0][colIdx].set_ylabel('Average\nfiring rate', fontsize=8)
            ax[0][colIdx].legend(fontsize=6)
            ax[1][colIdx].set_ylabel('Delay (ms)', fontsize=8)
        if delayYlim is not None:
            dYlim = [min(delayYlim[0], dYlim[0]), max(delayYlim[1], dYlim[1])]

    for colIdx in range(len(epochs)):
        adjust_lim(ax[1, :], dYlim, 'ylim')

    # delays = dict()
    # delays.update(
    #     {ascertain_colName_from_colParams(colParam):
    #          {key: delay for key, delay in zip([ascertain_colName_from_colParams(item) for item in traceConds],
    #                                            pick_delay(tms, selectBlocksinfo, zeroMTCond, traceConds,
    #                                                          activeNeu))}})
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
