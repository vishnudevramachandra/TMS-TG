import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import copy
from statistics import mean

from tms_tg import TMSTG, EPOCHISOLATORS
from cachetools import cached
from cachetools.keys import hashkey
from itertools import zip_longest
from figures.helper_figs import (adjust_lim, ascertain_colName_from_colParams, normalize_psfr, selectNeuron, fb,
                                 plot_MeanAndSEM, plot_populationAvgFR)


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_raster(tms, uniqueEpochs):
    return tms.psts


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_psfr(tms, uniqueEpochs):
    return tms.compute_firingrate


def make_subplots(colParams, rasterOnly, rasterAndPsfr, rasterAndPopulationAvgFR):
    if sum([rasterOnly, rasterAndPsfr, rasterAndPopulationAvgFR]) > 1:
        raise ValueError('Not correct truth values for plot Types')
    elif rasterOnly:
        return plt.subplots(3, len(colParams))
    else:
        return plt.subplots(4, len(colParams))


def plot(tms, colParams=None, rasterOnly=False, rasterAndPsfr=False, rasterAndPopulationAvgFR=False, xlim=None):
    assert type(colParams) == tuple or colParams is None, 'colNames has to be a list of dictionary items or None'
    rasterRowConds = ({'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '<1'}},
                      {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}},
                      {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '>1'}})
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2')

    if rasterOnly or rasterAndPsfr or rasterAndPopulationAvgFR:
        fig, ax = make_subplots(colParams, rasterOnly, rasterAndPsfr, rasterAndPopulationAvgFR)
        epochNumsAndNeuronNums_perCol = list()

        for colIdx in range(len(colParams)):
            colName = ascertain_colName_from_colParams(colParams[colIdx])
            tms.analysis_params = copy.deepcopy(colParams[colIdx])
            selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
            epochIndices = selectBlocksinfo.index.unique()

            if any(selectBlocksinfoIdx):
                # compute peristimulus timestamps and firing-rates
                ps_TS = compute_raster(tms, tuple(selectBlocksinfo.index.to_numpy()))
                activeNeu = tms.stats_is_signf_active()
                if rasterAndPsfr or rasterAndPopulationAvgFR:
                    # ps_FR, ps_T, ps_baseline_FR, _ = compute_psfr(tms, tuple(selectBlocksinfo.index.to_numpy()))
                    ps_FR, ps_T = tms.compute_firingrate(
                        *tms.analysis_params['peristimParams']['smoothingParams'].values(),
                        *tms.analysis_params['peristimParams']['timeWin'],
                        tms.analysis_params['peristimParams']['trigger'])
                    ps_baseline_FR, ps_baseline_T = (
                        tms.compute_firingrate('rectangular',
                                               np.diff(tms.analysis_params['peristimParams']['baselinetimeWin']).item(
                                                   0),
                                               0.0,
                                               mean(tms.analysis_params['peristimParams']['baselinetimeWin']),
                                               tms.analysis_params['peristimParams']['baselinetimeWin'][1],
                                               tms.analysis_params['peristimParams']['trigger']))
                    ps_T_corrected = tms.analysis_params['peristimParams']['timeWin'][0] + ps_T

                # randomly sample an epoch for plotting
                sampleEpochIndex = epochIndices[np.random.choice(len(epochIndices))]
                sampleBlocksinfo = selectBlocksinfo.loc[sampleEpochIndex, :]
                sampleActiveNeu = np.where(activeNeu[sampleEpochIndex])[0]

                # get the index of zeroMT ('MT' == 0) in order to exclude it from further selection (e.g., 'MT' <= 1)
                _, zeroMTIdx = fb(sampleBlocksinfo, zeroMTCond)

                # select peristimulus timestamps and firing-rates pertaining to sampled epoch
                blockIndices = np.where(selectBlocksinfo.index == sampleEpochIndex)[0]
                samplePSTS = [ps_TS[i] for i in blockIndices]
                if rasterAndPsfr or rasterAndPopulationAvgFR:
                    samplePSFR = [ps_FR[i] for i in blockIndices]
                    sampleBaselineFR = [ps_baseline_FR[i] for i in blockIndices]

                # statistics
                epochNumsAndNeuronNums_perCol.append(
                    (len(epochIndices), [len(ps_TS[np.where(selectBlocksinfo.index == epochIndex)[0][0]])
                                         for epochIndex in epochIndices]))

                # randomly select a neuron for plotting raster
                neuIdx = selectNeuron(sampleActiveNeu)

                # plot raster
                ax[0][colIdx].set_title(sampleEpochIndex[0] + '/' + colName + 'Neu-' + str(neuIdx), fontsize=6)
                for i in range(len(rasterRowConds)):
                    _, blockIdx = fb(sampleBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
                    assert sum(blockIdx) >= 1, 'something does not add up'

                    # if there are multiple blocks with same rasterRowCond, select the one with maximum no. of Trigs
                    if sum(blockIdx) > 1:
                        idx = np.where(blockIdx)[0][
                            sampleBlocksinfo.loc[blockIdx, 'no. of Trigs'].argmax()]
                        blockIdx.iloc[np.setdiff1d(np.where(blockIdx)[0], idx)] = False

                    ax[i][colIdx].eventplot([tms.analysis_params['peristimParams']['timeWin'][0] + item
                                             for item in samplePSTS[np.where(blockIdx)[0][0]][neuIdx]],
                                            colors=colorsPlt[i])
                    if colIdx == 0:
                        ax[i][colIdx].set_ylabel(
                            f'MT = {sampleBlocksinfo[blockIdx]["MT"].mean()}\nTrials (N)', fontsize=8)

                rowIdx = len(rasterRowConds)

                # plot PSFR
                if rasterAndPsfr:
                    for i in range(len(rasterRowConds)):
                        _, blockIdx = fb(sampleBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
                        intIndices = np.where(blockIdx)[0]
                        selectPSFR = samplePSFR[intIndices[0]]
                        for index in intIndices[1:]:
                            selectPSFR = np.append(selectPSFR, samplePSFR[index], axis=0)

                        plot_MeanAndSEM(selectPSFR.mean(axis=0)[:, neuIdx],
                                        selectPSFR.std(axis=0)[:, neuIdx] / np.sqrt(selectPSFR.shape[0]),
                                        ps_T_corrected,
                                        ax[rowIdx][colIdx],
                                        colorsPlt[i],
                                        f'MT = {sampleBlocksinfo[blockIdx]["MT"].mean()}')
                        if colIdx == 0:
                            ax[rowIdx][colIdx].set_ylabel('Firing rate (Hz)', fontsize=8)

                # plot population average firing rate
                if rasterAndPopulationAvgFR:
                    plot_populationAvgFR(tms, ps_T_corrected, ps_FR,
                                         selectBlocksinfo, rasterRowConds, zeroMTCond, activeNeu, ax, rowIdx,
                                         colIdx, colorsPlt)

        if xlim is not None:
            adjust_lim(ax, xlim, 'xlim')
        plt.show()
        print('[(Number of Epochs, Number of Neurons per epoch), ...]: ', epochNumsAndNeuronNums_perCol)


if __name__ == '__main__':
    nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))

    epochs = (
        {'selectionParams': {'Epoch': {'Region': 'MC', 'Layer': 'L5'}}},
        {'selectionParams': {'Epoch': {'Region': 'SC', 'Layer': 'L5'}}},
        {'selectionParams': {'Epoch': {'Region': 'thal', 'Layer': None}, 'RecArea ': 'BZ'}},
        {'selectionParams': {'Epoch': {'Region': 'thal', 'Layer': None}, 'RecArea ': 'CZ'}},
    )

    epochs = (
        {'selectionParams': {'Epoch': {'Region': 'MC', 'Layer': 'L5'}}},
        {'selectionParams': {'Epoch': {'Region': 'SC', 'Layer': 'L5'}}},
        {'selectionParams': {'Epoch': {'Region': 'VC', 'Layer': 'L5'}}},
    )

    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    # plot(tms, colParams=epochs, rasterOnly=False, rasterAndPsfr=False, rasterAndPopulationAvgFR=True, xlim=[-20, 60])
    plot(tms, colParams=epochs, rasterOnly=False, rasterAndPsfr=True, rasterAndPopulationAvgFR=False, xlim=[-20, 60])
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
