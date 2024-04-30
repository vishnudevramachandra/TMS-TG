import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import copy
import os
import seaborn as sns

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import adjust_lim, plot_populationAvgFR, pick_delay, gb_addinfo
from lib.helper_tms_tg import merge_selectionParams


def plot(tms, activeNeus, colParam, axes, xlim=None, frKind=None, frYlim=None, delayYlim=None):
    remSelectionParams = EPOCHISOLATORS.copy()
    remSelectionParams.pop(2)
    traceConds = ({'selectionParams': {'Epoch': {'Layer': 'L23', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L4', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L5', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L6', **dict(zip_longest(remSelectionParams, [None, ]))}}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    postConds = [{'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), postKind: 'Post'}} for
                 postKind in {'Skin-Injection', 'TG-Injection ', 'TGOrifice'} & set(tms.blocksinfo.columns)]
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2', 'C3')

    if not any(selectBlocksinfoIdx):
        print(f'Cannot plot for {colParam} as the associated data is missing in this group of animals')
        return frYlim, delayYlim

    # --------------------------------------------------------------------------------------------------------------
    # add a title
    colName = colParam['selectionParams']['Epoch']['Region']
    axes[0].set_title(colName, fontsize=8)

    # --------------------------------------------------------------------------------------------------------------
    delays = list()
    oldSelParams = copy.deepcopy(tms.analysis_params['selectionParams'])
    for idx, traceCond in enumerate(traceConds):
        tms.analysis_params = {'selectionParams': merge_selectionParams(
            copy.deepcopy(tms.analysis_params['selectionParams']), copy.deepcopy(traceCond['selectionParams']))}

        # plot population activity
        plot_populationAvgFR(tms, activeNeus, ax=axes[0], kind=frKind, excludeConds=[zeroMTCond, *postConds],
                             lineLabel=traceCond['selectionParams']['Epoch']['Layer'], lineColor=colorsPlt[idx])

        # compute delays
        delays.append(pick_delay(tms, activeNeus, excludeConds=[zeroMTCond, *postConds]))

        # reset analysis parameters
        tms.analysis_params = {'selectionParams': oldSelParams}

    # plot delay
    sns.swarmplot(data=delays, color='k', size=2, ax=axes[1])
    sns.violinplot(data=delays, inner=None, cut=1, bw_adjust=0.5, ax=axes[1])
    axes[1].set_xticks(range(len(traceConds)),
                       labels=[item['selectionParams']['Epoch']['Layer'] for item in traceConds])

    # --------------------------------------------------------------------------------------------------------------
    if xlim is not None:
        adjust_lim(axes[0], xlim, 'xlim')

    return axes[0].get_ylim(), axes[1].get_ylim()


if __name__ == '__main__':
    # nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    colParams = (
        {'selectionParams': {'Epoch': {'Region': 'MC', }, 'MT': '==1'}},
        {'selectionParams': {'Epoch': {'Region': 'SC', }, 'MT': '==1'}},
        {'selectionParams': {'Epoch': {'Region': 'VC', }, 'MT': '==1'}},
    )
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = pd.read_pickle('../../grandBlocksinfo')
        for col in {'psActivity', 'delay'} & set(grandBlocksinfo.columns):
            bIndicesInGBIndices = tuple(set(tms.blocksinfo.index.unique()) & set(grandBlocksinfo.index.unique()))
            for index in bIndicesInGBIndices:
                tms.blocksinfo.loc[index, col] = grandBlocksinfo.loc[index, col]
        tms.blocksinfo = tms.blocksinfo.where(pd.notnull(tms.blocksinfo), None)
        tms.filter_blocks = None
    # activeNeus = tms.stats_is_signf_active()
    activeNeus = pd.read_pickle("./activeNeu_Cortex")

    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()
    dYlim = [np.inf, 0]
    fig, ax = plt.subplots(2, len(colParams))
    for colIdx in range(len(colParams)):
        tms.analysis_params = copy.deepcopy(colParams[colIdx])
        selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
        epochIndices = selectBlocksinfo.index.unique()

        # statistics
        animalNumsEpochNumsAndActiveNeuronNums_perCol.append((np.unique([item[0] for item in epochIndices]).size,
                                                              len(epochIndices),
                                                              [activeNeus[item].sum() for item in epochIndices]))
        # do the plots
        frYlim, delayYlim = plot(
            tms, activeNeus, colParams[colIdx], ax[:, colIdx], xlim=[-25, 50], frKind='Average')
        if colIdx == 0:
            ax[0][colIdx].set_ylabel('Average\nfiring rate', fontsize=8)
            ax[0][colIdx].legend(fontsize=6)
            ax[1][colIdx].set_ylabel('Delay (ms)', fontsize=8)
        if delayYlim is not None:
            dYlim = [min(delayYlim[0], dYlim[0]), max(delayYlim[1], dYlim[1])]

    for colIdx in range(len(colParams)):
        adjust_lim(ax[1, :], dYlim, 'ylim')

    plt.savefig('figure1b.pdf', dpi='figure', format='pdf')
    plt.show()
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums_perCol)

    # save data after join and apply
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = gb_addinfo(pd.read_pickle('../../grandBlocksinfo'), tms)
        grandBlocksinfo.to_pickle('../../grandBlocksinfo')

    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
