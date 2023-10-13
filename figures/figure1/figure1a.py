import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import copy

from tms_tg import TMSTG, FilterBlocks, EPOCHISOLATORS
from cachetools import cached
from cachetools.keys import hashkey
from itertools import zip_longest


def adjust_xlim(ax, xlim):
    for axes in np.nditer(ax, flags=['refs_ok']):
        axes.item().set_xlim(xlim)


def colName_from_dict(colName, params):
    for key in params:
        if type(params[key]) == dict:
            colName = colName_from_dict(colName, params[key])
        elif params[key] is not None:
            colName += params[key] + '/'
    return colName


def ascertain_colName_from_colParams(colParams):
    if colParams is None:
        colName = 'All'
    else:
        colName = colName_from_dict('', colParams)
    return colName


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_raster(tms, uniqueEpochs):
    return tms.psts


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_psfr(tms, uniqueEpochs):
    return tms.psfr


def normalize_psfr(tms, fb, ps_T, ps_FR, blocksInfo):
    LATE_COMP_TIMEWIN = (10, 50)
    PARAMS = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}}
    _, mtBlocksIdx = fb(blocksInfo, PARAMS)
    ps_T = ps_T + tms.analysis_params['peristimParams']['timeWin'][0]
    normalizedPSFR = [None] * len(ps_FR)

    for epochIndex in blocksInfo.index.unique():
        boolEpochIndex = blocksInfo.index == epochIndex

        # take PSFR at motor threshold
        mtIndices = np.where(boolEpochIndex & mtBlocksIdx.to_numpy())[0]
        mtPSFR = ps_FR[mtIndices[0]]
        for index in mtIndices[1:]:
            mtPSFR = np.append(mtPSFR, ps_FR[index], axis=0)
        avg_mtPSFR = mtPSFR.mean(axis=0, keepdims=True)
        peakFR_latecomp = avg_mtPSFR[:,
                        (LATE_COMP_TIMEWIN[0] <= ps_T) & (ps_T < LATE_COMP_TIMEWIN[1]),
                        :].max(axis=1, keepdims=True)

        # take the peak firing rate of the late component for PSFR at motor threshold and do the normalization
        for i in np.where(boolEpochIndex)[0]:
            normalizedPSFR[i] = ps_FR[i] / peakFR_latecomp

    return normalizedPSFR


def selectNeuron(ps_TS):
    num_of_neurons = [len(item) for item in ps_TS]
    num_of_spikes = np.array([sum([len(row) for row in neuron])
                              for item in ps_TS for neuron in item]).reshape(len(ps_TS),
                                                                             sum(num_of_neurons) // len(ps_TS))
    return np.random.choice(np.flatnonzero(num_of_spikes.sum(axis=0)))


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
    fb = FilterBlocks()

    if rasterOnly or rasterAndPsfr or rasterAndPopulationAvgFR:
        fig, ax = make_subplots(colParams, rasterOnly, rasterAndPsfr, rasterAndPopulationAvgFR)
        epochNumsAndNeuronNums_perCol = list()

        for j in range(len(colParams)):
            colName = ascertain_colName_from_colParams(colParams[j])
            tms.analysis_params = copy.deepcopy(colParams[j])
            selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
            epochIndices = selectBlocksinfo.index.unique()

            if any(selectBlocksinfoIdx):
                # compute peristimulus timestamps and firing-rates
                ps_TS = compute_raster(tms, tuple(selectBlocksinfo.index.to_numpy()))
                if rasterAndPsfr or rasterAndPopulationAvgFR:
                    ps_FR, ps_T, ps_baseline_FR, _ = compute_psfr(tms, tuple(selectBlocksinfo.index.to_numpy()))
                    ps_T_corrected = tms.analysis_params['peristimParams']['timeWin'][0] + ps_T

                # randomly sample an epoch for plotting
                sampleEpochIndex = epochIndices[np.random.choice(len(epochIndices))]
                sampleBlocksinfo = selectBlocksinfo.loc[sampleEpochIndex, :]

                # get the index of zeroMT ('MT' == 0) in order to exclude it from further selection (e.g., 'MT' <= 1)
                _, zeroMTIdx = fb(sampleBlocksinfo, zeroMTCond)

                # select peristimulus timestamps and firing-rates pertaining to sampled epoch
                indices = np.where(selectBlocksinfo.index == sampleEpochIndex)[0]
                samplePSTS = [ps_TS[i] for i in indices]
                if rasterAndPsfr or rasterAndPopulationAvgFR:
                    samplePSFR = [ps_FR[i] for i in indices]
                    sampleBaselineFR = [ps_baseline_FR[i] for i in indices]

                # statistics
                epochNumsAndNeuronNums_perCol.append(
                    (len(epochIndices), [len(ps_TS[np.where(selectBlocksinfo.index == epochIndex)[0][0]])
                                         for epochIndex in epochIndices]))

                # randomly select a neuron for plotting raster
                neuIdx = selectNeuron([samplePSTS[i] for i in np.where(~zeroMTIdx)[0]])

                # plot raster
                ax[0][j].set_title(sampleEpochIndex[0] + '/' + colName + 'Neu-' + str(neuIdx), fontsize=6)
                for i in range(len(rasterRowConds)):
                    _, rowSelectBlockIdx = fb(sampleBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
                    assert sum(rowSelectBlockIdx) >= 1, 'something does not add up'

                    # if there are multiple blocks with same rasterRowCond, select the one with maximum no. of Trigs
                    if sum(rowSelectBlockIdx) > 1:
                        idx = np.where(rowSelectBlockIdx)[0][
                            sampleBlocksinfo.loc[rowSelectBlockIdx, 'no. of Trigs'].argmax()]
                        rowSelectBlockIdx.iloc[np.setdiff1d(np.where(rowSelectBlockIdx)[0], idx)] = False

                    ax[i][j].eventplot([tms.analysis_params['peristimParams']['timeWin'][0] + item
                                        for item in samplePSTS[np.where(rowSelectBlockIdx)[0][0]][neuIdx]],
                                       colors=colorsPlt[i])

                # plot PSFR
                if rasterAndPsfr:
                    for i in range(len(rasterRowConds)):
                        _, rowSelectBlockIdx = fb(sampleBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
                        indices = np.where(rowSelectBlockIdx)[0]
                        selectPSFR = samplePSFR[indices[0]]
                        for index in indices[1:]:
                            selectPSFR = np.append(selectPSFR, samplePSFR[index], axis=0)
                        selectPSFR_mean = selectPSFR.mean(axis=0)[:, neuIdx]
                        selectPSFR_sem = selectPSFR.std(axis=0)[:, neuIdx] / np.sqrt(selectPSFR.shape[0])
                        ax[3][j].plot(ps_T_corrected, selectPSFR_mean, color=colorsPlt[i])
                        ax[3][j].fill_between(
                            ps_T_corrected, selectPSFR_mean - selectPSFR_sem, selectPSFR_mean + selectPSFR_sem,
                            alpha=0.2, color=colorsPlt[i])

                # plot population average firing rate
                if rasterAndPopulationAvgFR:
                    normalizedPSFR = normalize_psfr(tms, fb, ps_T, ps_FR, selectBlocksinfo)
                    _, zeroMTIdx = fb(selectBlocksinfo, zeroMTCond)
                    for i in range(len(rasterRowConds)):
                        _, rowSelectBlockIdx = fb(selectBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
                        indices = np.where(rowSelectBlockIdx)[0]
                        avgNormalizedPSFR = normalizedPSFR[indices[0]].mean(axis=0)
                        for index in indices[1:]:
                            avgNormalizedPSFR = np.append(avgNormalizedPSFR,
                                                          normalizedPSFR[index].mean(axis=0), axis=1)
                        avgNormalizedPSFR_mean = avgNormalizedPSFR.mean(axis=1)
                        avgNormalizedPSFR_sem = avgNormalizedPSFR.std(axis=1) / np.sqrt(avgNormalizedPSFR.shape[1])
                        ax[3][j].plot(ps_T_corrected, avgNormalizedPSFR_mean, color=colorsPlt[i])
                        ax[3][j].fill_between(ps_T_corrected,
                                              avgNormalizedPSFR_mean - avgNormalizedPSFR_sem,
                                              avgNormalizedPSFR_mean + avgNormalizedPSFR_sem,
                                              alpha=0.2, color=colorsPlt[i])

        if xlim is not None:
            adjust_xlim(ax, xlim)
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
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    plot(tms, colParams=epochs, rasterOnly=False, rasterAndPsfr=False, rasterAndPopulationAvgFR=True, xlim=[-20, 60])

    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
