import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import copy
import pickle

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import adjust_lim, normalize_psfr, plot_populationAvgFR, fb, plot_delay


def plot(tms, activeNeu, colParams=None, xlim=None, kind=None):
    remSelectionParams = EPOCHISOLATORS.copy()
    remSelectionParams.pop(2)
    traceConds = ({'selectionParams': {'Epoch': {'Layer': 'L23', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L4', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L5', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L6', **dict(zip_longest(remSelectionParams, [None, ]))}}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2', 'C3')
    ylim = [np.inf, 0]
    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()
    delays = dict()
    fig, ax = plt.subplots(2, len(colParams))

    for colIdx in range(len(colParams)):
        colName = colParams[colIdx]['selectionParams']['Epoch']['Region']
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
            animalNumsEpochNumsAndActiveNeuronNums_perCol.append((np.unique([item[0] for item in epochIndices]).size,
                                                                  len(epochIndices),
                                                                  [activeNeu[item].sum() for item in epochIndices]))

            ax[0][colIdx].set_title(colName, fontsize=8)

            # plot population activity
            match kind:
                case 'Normalized':
                    plot_populationAvgFR(normalize_psfr(tms, fb, ps_T_corrected, ps_FR, selectBlocksinfo),
                                         ps_T_corrected, selectBlocksinfo, traceConds, zeroMTCond, activeNeu,
                                         ax[0][colIdx], colorsPlt,
                                         [item['selectionParams']['Epoch']['Layer'] for item in traceConds])
                    if colIdx == 0:
                        ax[0][colIdx].set_ylabel('Normalized\nfiring rate', fontsize=8)
                case _:
                    plot_populationAvgFR(ps_FR,
                                         ps_T_corrected, selectBlocksinfo, traceConds, zeroMTCond, activeNeu,
                                         ax[0][colIdx], colorsPlt,
                                         [item['selectionParams']['Epoch']['Layer'] for item in traceConds])
                    if colIdx == 0:
                        ax[0][colIdx].set_ylabel('Average\nfiring rate', fontsize=8)

            if colIdx == 0:
                ax[0][colIdx].legend(fontsize=6)

            # plot delay
            plot_delay(delays, ax, colParams, colIdx, tms, selectBlocksinfo, zeroMTCond, traceConds, activeNeu,
                       swarmplotsize=1)
            ax[1][colIdx].set_xticks([0, 1, 2, 3],
                                     labels=[item['selectionParams']['Epoch']['Layer'] for item in traceConds])

            ylim = [min(ax[1][colIdx].get_ylim()[0], ylim[0]), max(ax[1][colIdx].get_ylim()[1], ylim[1])]
            if colIdx == 0:
                ax[1][colIdx].set_ylabel('Delay (ms)', fontsize=8)

    if xlim is not None:
        adjust_lim(ax[0, :], xlim, 'xlim')
    adjust_lim(ax[1, :], ylim, 'ylim')
    plt.savefig('figure1b.pdf', dpi='figure', format='pdf')
    plt.show()
    with open('delays.pickle', 'wb') as f:
        pickle.dump(delays, f)
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums_perCol)


if __name__ == '__main__':
    # nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    epochs = (
        {'selectionParams': {'Epoch': {'Region': 'MC', }, 'MT': '==1'}},
        {'selectionParams': {'Epoch': {'Region': 'SC', }, 'MT': '==1'}},
        {'selectionParams': {'Epoch': {'Region': 'VC', }, 'MT': '==1'}},
    )
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    activeNeu = tms.stats_is_signf_active()
    activeNeu = pd.read_pickle("./activeNeu_Cortex")
    plot(tms, activeNeu, colParams=epochs, xlim=[-20, 60])

    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
