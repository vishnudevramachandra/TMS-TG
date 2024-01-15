import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import copy

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import adjust_lim, plot_populationAvgFR, fb, plot_delay


def plot(tms, activeNeu, colParams=None, xlim=None):
    remSelectionParams = EPOCHISOLATORS.copy()
    remSelectionParams.pop(2)
    traceConds = ({'selectionParams': {'Epoch': {'Layer': 'L23', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L4', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L5', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L6', **dict(zip_longest(remSelectionParams, [None, ]))}}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2', 'C3')
    epochNumsAndActiveNeuronNums_perCol = list()
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
            epochNumsAndActiveNeuronNums_perCol.append(
                (len(epochIndices), [activeNeu[epochIndex].sum() for epochIndex in epochIndices]))

            ax[0][colIdx].set_title(colName, fontsize=8)
            # plot population activity
            plot_populationAvgFR(ps_FR, ps_T_corrected,
                                 selectBlocksinfo, traceConds, zeroMTCond, activeNeu, ax[0][colIdx],
                                 colorsPlt, [item['selectionParams']['MT'] for item in traceConds])
            if colIdx == 0:
                ax[0][colIdx].legend(fontsize=6)

            # plot delay
            delays = list()
            meanPSFR, t, meanBaselineFR, _ = tms.avg_FR_per_neuron(squeezeDim=False)
            _, zeroMTIdx = fb(selectBlocksinfo, zeroMTCond)
            for i in range(len(traceConds)):
                blocksInfo, blockIdx = fb(selectBlocksinfo, traceConds[i], ~zeroMTIdx)
                if blockIdx.sum() > 0:
                    selectMeanPSFR = np.concatenate([meanPSFR[n][:, :, activeNeu[item]][0, :, :]
                                                     for n, item in zip(np.flatnonzero(blockIdx), blocksInfo.index)],
                                                    axis=1)
                    tmp = tms.late_comp.compute_delay(selectMeanPSFR,
                                                      t,
                                                      selectMeanPSFR[t < 0, :].max(axis=0, keepdims=True),
                                                      tms.analysis_params['peristimParams']['smoothingParams'][
                                                          'width'] + 0.25)
                    if np.isnan(tmp).sum() != len(tmp):
                        delays.append(tmp[~np.isnan(tmp)])
                    else:
                        delays.append(np.array([np.nan, np.nan]))

                else:
                    delays.append(np.array([np.nan, np.nan]))

            vplot = ax[1][colIdx].violinplot(delays, showmeans=True)
            ax[1][colIdx].set_xticks([1, 2, 3, 4],
                                     labels=[item['selectionParams']['Epoch']['Layer'] for item in traceConds])
            for pc, color in zip(vplot['bodies'], colorsPlt):
                pc.set_facecolor(color)

            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                vplot[partname].set_colors(colorsPlt)

            if colIdx == 0:
                ax[1][colIdx].set_ylabel('Delay (ms)', fontsize=8)

    if xlim is not None:
        adjust_lim(ax[0, :], xlim, 'xlim')
    adjust_lim(ax[1, :], ylim, 'ylim')
    plt.show()
    print('[(Number of Epochs, Number of Neurons per epoch), ...]: ', epochNumsAndActiveNeuronNums_perCol)


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
    # activeNeu = tms.stats_is_signf_active()
    activeNeu = pd.read_pickle("./activeNeu_Cortex")
    plot(tms, activeNeu, colParams=epochs, xlim=[-20, 60])

    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
