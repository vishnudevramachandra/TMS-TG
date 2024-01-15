import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import copy
import pickle

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import (adjust_lim, plot_populationAvgFR, fb, normalize_psfr, plot_delay)


def plot(tms, activeNeu, colParams=None, xlim=None, kind=None):
    traceConds = ({'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '<1'}},
                  {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}},
                  {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '>1'}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2')
    ylim = [np.inf, 0]
    epochNumsAndActiveNeuronNums_perCol = list()
    delays = dict()
    fig, ax = plt.subplots(2, len(colParams))

    for colIdx in range(len(colParams)):
        colName = colParams[colIdx]['selectionParams']['RecArea ']
        tms.analysis_params = copy.deepcopy(colParams[colIdx])
        selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
        epochIndices = selectBlocksinfo.index.unique()

        if any(selectBlocksinfoIdx):
            ps_FR, ps_T = tms.compute_firingrate(
                *tms.analysis_params['peristimParams']['smoothingParams'].values(),
                *tms.analysis_params['peristimParams']['timeWin'],
                tms.analysis_params['peristimParams']['trigger'])
            ps_T_corrected = tms.analysis_params['peristimParams']['timeWin'][0] + ps_T

            # statistics
            epochNumsAndActiveNeuronNums_perCol.append(
                (len(epochIndices), [activeNeu[epochIndex].sum() for epochIndex in epochIndices]))

            ax[0][colIdx].set_title(colName, fontsize=8)

            # plot population activity
            match kind:
                case 'Normalized':
                    plot_populationAvgFR(normalize_psfr(tms, fb, ps_T_corrected, ps_FR, selectBlocksinfo),
                                         ps_T_corrected, selectBlocksinfo, traceConds, zeroMTCond, activeNeu,
                                         ax[0][colIdx], colorsPlt,
                                         [item['selectionParams']['MT'] for item in traceConds])
                    if colIdx == 0:
                        ax[0][colIdx].set_ylabel('Normalized\nfiring rate', fontsize=8)
                case _:
                    plot_populationAvgFR(ps_FR, ps_T_corrected,
                                         selectBlocksinfo, traceConds, zeroMTCond, activeNeu, ax[0][colIdx],
                                         colorsPlt, [item['selectionParams']['MT'] for item in traceConds])
                    if colIdx == 0:
                        ax[0][colIdx].set_ylabel('Average\nfiring rate', fontsize=8)

            if colIdx == 0:
                ax[0][colIdx].legend(fontsize=6)

            # plot delay
            plot_delay(delays, ax, colParams, colIdx, tms, selectBlocksinfo, zeroMTCond, traceConds, activeNeu)

            if colIdx == 0:
                ax[1][colIdx].set_ylabel('Delay (ms)', fontsize=8)
            ylim = [min(ax[1][colIdx].get_ylim()[0], ylim[0]), max(ax[1][colIdx].get_ylim()[1], ylim[1])]

    if xlim is not None:
        adjust_lim(ax[0, :], xlim, 'xlim')
    adjust_lim(ax[1, :], ylim, 'ylim')
    plt.savefig('figure2.pdf', dpi='figure', format='pdf')
    plt.show()
    with open('delays.pickle', 'wb') as f:
        pickle.dump(delays, f)
    print('[(Number of Epochs, Number of Neurons per epoch), ...]: ', epochNumsAndActiveNeuronNums_perCol)


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
    activeNeu = pd.read_pickle("./activeNeu_thal")
    plot(tms, activeNeu, colParams=epochs, xlim=[-20, 60], kind='Normalized')

    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
