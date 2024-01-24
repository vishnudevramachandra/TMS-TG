import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import copy
import pandas as pd
from statistics import mean

from tms_tg import TMSTG, EPOCHISOLATORS
from cachetools import cached
from cachetools.keys import hashkey
from itertools import zip_longest
from figures.helper_figs import (adjust_lim, ascertain_colName_from_colParams, selectEpochAndNeuron, fb,
                                 plot_MeanAndSEM, plot_populationAvgFR)


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_raster(tms, uniqueEpochs):
    return tms.psts


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_psfr(tms, uniqueEpochs):
    return tms.compute_firingrate


def construct_subplots(colParams, kind):
    match kind:
        case 'rasterOnly':
            return plt.subplots(3, len(colParams))
        case 'rasterAndPsfr' | 'rasterAndPopulationAvgFR':
            return plt.subplots(4, len(colParams))


def plot(tms, activeNeus, kind=None, colParams=None, xlim=None, epochAndNeuron=None):
    assert type(colParams) == tuple or colParams is None, 'colNames has to be a list of dictionary items or None'
    assert kind in ('rasterOnly', 'rasterAndPsfr', 'rasterAndPopulationAvgFR'), 'unknown input for parameter "kind"'
    rasterRowConds = ({'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '<1'}},
                      {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}},
                      {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '>1'}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2')
    ylim = [np.inf, 0]
    fig, ax = construct_subplots(colParams, kind)
    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()

    for colIdx in range(len(colParams)):
        colName = ascertain_colName_from_colParams(colParams[colIdx])
        tms.analysis_params = copy.deepcopy(colParams[colIdx])
        selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
        epochIndices = selectBlocksinfo.index.unique()

        if any(selectBlocksinfoIdx):
            # select an epoch and a neuron for plotting
            sampleBlocksinfo, sampleEpochIndex, neuIdx = (
                selectEpochAndNeuron(None if epochAndNeuron is None else epochAndNeuron[colIdx],
                                     tms, epochIndices, activeNeus, colParams[colIdx]))

            # get the index of zeroMT ('MT' == 0) in order to exclude it from further selection (e.g., 'MT' <= 1)
            _, zeroMTIdx = fb(sampleBlocksinfo, zeroMTCond)

            # select peristimulus timestamps and firing-rates pertaining to sampled epoch
            samplePSTS = compute_raster(tms, tuple(sampleBlocksinfo.index.to_numpy()))
            if kind in ('rasterAndPsfr', 'rasterAndPopulationAvgFR'):
                samplePSFR, ps_T = tms.compute_firingrate(
                    *tms.analysis_params['peristimParams']['smoothingParams'].values(),
                    *tms.analysis_params['peristimParams']['timeWin'],
                    tms.analysis_params['peristimParams']['trigger'])
                sampleBaselineFR, ps_baseline_T = (
                    tms.compute_firingrate('rectangular',
                                           np.diff(tms.analysis_params['peristimParams']['baselinetimeWin']).item(
                                               0),
                                           0.0,
                                           mean(tms.analysis_params['peristimParams']['baselinetimeWin']),
                                           tms.analysis_params['peristimParams']['baselinetimeWin'][1],
                                           tms.analysis_params['peristimParams']['trigger']))
                ps_T_corrected = tms.analysis_params['peristimParams']['timeWin'][0] + ps_T

            # statistics
            animalNumsEpochNumsAndActiveNeuronNums_perCol.append((np.unique([item[0] for item in epochIndices]).size,
                                                                  len(epochIndices),
                                                                  [activeNeus[item].sum() for item in epochIndices]))

            rowIdx = len(rasterRowConds)

            # plot raster
            ax[0][colIdx].set_title(sampleEpochIndex[0] + '/' + colName + 'Neu-' + str(neuIdx), fontsize=6)
            for i in range(len(rasterRowConds)):
                _, blockIdx = fb(sampleBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
                assert sum(blockIdx) >= 1, \
                    f"epoch {sampleEpochIndex} does not have MT{rasterRowConds[i]['selectionParams']['MT']} values"

                # if there are multiple blocks with same rasterRowCond, select the one with maximum no. of Trigs
                if sum(blockIdx) > 1:
                    idx = np.where(blockIdx)[0][sampleBlocksinfo.loc[blockIdx, 'no. of Trigs'].argmax()]
                    blockIdx.iloc[np.setdiff1d(np.where(blockIdx)[0], idx)] = False

                ax[i][colIdx].eventplot([tms.analysis_params['peristimParams']['timeWin'][0] + item
                                         for item in samplePSTS[np.nonzero(blockIdx)[0][0]][neuIdx]],
                                        colors=colorsPlt[i])
                if colIdx == 0:
                    ax[i][colIdx].set_ylabel(
                        f'MT = {sampleBlocksinfo[blockIdx]["MT"].values[0]}\nTrials (N)', fontsize=8)

                # plot PSFR
                if kind == 'rasterAndPsfr':
                    selectPSFR = samplePSFR[np.nonzero(blockIdx)[0][0]]
                    plot_MeanAndSEM(selectPSFR.mean(axis=0)[:, neuIdx],
                                    selectPSFR.std(axis=0)[:, neuIdx] / np.sqrt(selectPSFR.shape[0]),
                                    ps_T_corrected,
                                    ax[rowIdx][colIdx],
                                    colorsPlt[i],
                                    f'MT = {sampleBlocksinfo[blockIdx]["MT"].mean()}')
                    if colIdx == 0:
                        ax[rowIdx][colIdx].set_ylabel('Firing rate (Hz)', fontsize=8)

                # plot population average firing rate
                if kind == 'rasterAndPopulationAvgFR':
                    plot_populationAvgFR(samplePSFR,
                                         ps_T_corrected, selectBlocksinfo, rasterRowConds, zeroMTCond, activeNeus,
                                         ax[rowIdx][colIdx], colorsPlt,
                                         [item['selectionParams']['MT'] for item in rasterRowConds])

            if kind != 'rasterOnly':
                ylim = [min(ax[rowIdx][colIdx].get_ylim()[0], ylim[0]), max(ax[rowIdx][colIdx].get_ylim()[1], ylim[1])]

    if xlim is not None:
        adjust_lim(ax, xlim, 'xlim')
    # if kind != 'rasterOnly':
    #     adjust_lim(ax[rowIdx, :], ylim, 'ylim')
    plt.savefig('figure1a.pdf', dpi='figure', format='pdf')
    plt.show()
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums_perCol)


if __name__ == '__main__':
    nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))

    epochs = (
        {'selectionParams': {'Epoch': {'Region': 'MC', 'Layer': 'L5'}}},
        {'selectionParams': {'Epoch': {'Region': 'SC', 'Layer': 'L5'}}},
        {'selectionParams': {'Epoch': {'Region': 'thal', 'Layer': None}, 'RecArea ': 'BZ'}},
        {'selectionParams': {'Epoch': {'Region': 'thal', 'Layer': None}, 'RecArea ': 'CZ'}},
    )
    keys = ('Region', 'Layer', 'Animal', 'Mov', 'Depth', 'CoilHemVsRecHem')
    epochs = (
        {'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('MC', 'L5'))}}},
        {'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('SC', 'L5'))}}},
        {'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('VC', 'L5'))}}},
    )

    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    tms.analysis_params = {'peristimParams': {'timeWin': (-50.0, 350.0)}}
    # activeNeu = tms.stats_is_signf_active()
    activeNeu = pd.read_pickle("./activeNeu_Cortex")
    # plot(tms, colParams=epochs, rasterOnly=False, rasterAndPsfr=False, rasterAndPopulationAvgFR=True, xlim=[-20, 60])
    plotKinds = ('rasterOnly', 'rasterAndPsfr', 'rasterAndPopulationAvgFR')
    plot(tms, activeNeu, kind=plotKinds[1], colParams=epochs, xlim=[-50, 350], epochAndNeuron=
         ({'epochIndex': ('20200705', 'MC', 'L5', 'opposite', 'none', '1467'), 'neuIdx': 5},
          {'epochIndex': ('20200705', 'SC', 'L5', 'same', 'none', '1496'), 'neuIdx': 4},
          {'epochIndex': ('20200705', 'VC', 'L5', 'same', 'none', '1404'), 'neuIdx': 1}))
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
