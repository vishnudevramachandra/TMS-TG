import numpy as np
from itertools import zip_longest
import seaborn as sns
import copy

from tms_tg import EPOCHISOLATORS, FilterBlocks


fb = FilterBlocks()


def adjust_lim(ax, lim, kind):
    for axes in np.nditer(ax, flags=['refs_ok']):
        match kind:
            case 'xlim':
                axes.item().set_xlim(lim)
            case 'ylim':
                axes.item().set_ylim(lim)


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


def normalize_psfr(tms, fb, ps_T_corrected, ps_FR, blocksInfo):
    LATE_COMP_TIMEWIN = (5, 50)  # in msecs
    PARAMS = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}}
    _, mtBlocksIdx = fb(blocksInfo, PARAMS)
    normalizedPSFR = [None] * len(ps_FR)

    for epochIndex in blocksInfo.index.unique():
        boolEpochIndex = blocksInfo.index == epochIndex

        # take PSFR at motor threshold
        mtIndices = np.where(boolEpochIndex & mtBlocksIdx.to_numpy())[0]
        mtPSFR = ps_FR[mtIndices[0]]
        for index in mtIndices[1:]:
            mtPSFR = np.append(mtPSFR, ps_FR[index], axis=0)
        avg_mtPSFR = mtPSFR.mean(axis=0, keepdims=True)
        peakFR_latecomp = \
            (avg_mtPSFR[:, (LATE_COMP_TIMEWIN[0] <= ps_T_corrected) & (ps_T_corrected < LATE_COMP_TIMEWIN[1]), :].
             max(axis=1, keepdims=True))

        # take the peak firing rate of the late component for PSFR at motor threshold and do the normalization
        for i in np.where(boolEpochIndex)[0]:
            normalizedPSFR[i] = ps_FR[i] / peakFR_latecomp

    return normalizedPSFR


def selectNeuron(activeNeu):
    return np.random.choice(activeNeu)


def selectEpochAndNeuron(epochAndNeuron, tms, epochIndices, activeNeus, colParam):
    # randomly sample an epoch that has at least one active neuron for plotting if EpochAndNeuron is None
    sampleActiveNeus = [False, ]
    while not any(sampleActiveNeus):
        sampleEpochIndex = epochIndices[np.random.choice(len(epochIndices))] \
            if (epochAndNeuron is None or 'epochIndex' not in epochAndNeuron.keys()) else epochAndNeuron['epochIndex']
        sampleParams = copy.deepcopy(colParam)
        sampleParams['selectionParams']['Epoch'] = {key: value
                                                    for key, value in zip(epochIndices.names, sampleEpochIndex)}
        tms.analysis_params = sampleParams
        sampleActiveNeus = activeNeus[sampleEpochIndex]
        if epochAndNeuron is not None and 'epochIndex' in epochAndNeuron.keys():
            assert any(sampleActiveNeus), f'Epoch {sampleEpochIndex} does not have any active neurons'
            break

    # randomly select a neuron for plotting raster
    neuIdx = np.random.choice(sampleActiveNeus.nonzero()[0]) \
        if epochAndNeuron is None or 'neuIdx' not in epochAndNeuron.keys() else epochAndNeuron['neuIdx']

    sampleBlocksinfo, _ = tms.filter_blocks

    return sampleBlocksinfo, sampleEpochIndex, neuIdx


def plot_MeanAndSEM(meanPSFR, semPSFR, t, ax, pltColor, lineLabel):
    ax.plot(t, meanPSFR, color=pltColor, label=lineLabel)
    ax.fill_between(t, meanPSFR - semPSFR, meanPSFR + semPSFR, alpha=0.2, color=pltColor)


def exclude_corrupt_traces(avgPSActivity):
    return np.delete(avgPSActivity, avgPSActivity.max(axis=0) > 20, axis=1)


def plot_populationAvgFR(psActivity, ps_T_corrected, selectBlocksinfo, traceConds, zeroMTCond, activeNeu,
                         ax, colorsPlt, labels=None):
    _, zeroMTIdx = fb(selectBlocksinfo, zeroMTCond)

    for i in range(len(traceConds)):
        _, blockIdx = fb(selectBlocksinfo, traceConds[i], ~zeroMTIdx)
        intIndices = np.where(blockIdx)[0]
        if len(intIndices) >= 1:
            index = intIndices[0]
            avgPSActivity = psActivity[index][:, :, (activeNeu[blockIdx.index[index]]
                                                     if activeNeu[blockIdx.index[index]].size > 1
                                                     else activeNeu[blockIdx.index[index]][np.newaxis])].mean(axis=0)
            for index in intIndices[1:]:
                boolIndex = activeNeu[blockIdx.index[index]] if activeNeu[blockIdx.index[index]].size > 1 \
                    else activeNeu[blockIdx.index[index]][np.newaxis]
                avgPSActivity = np.append(avgPSActivity,
                                          psActivity[index][:, :, boolIndex].mean(axis=0),
                                          axis=1)

            # avgPSActivity = exclude_corrupt_traces(avgPSActivity)
            plot_MeanAndSEM(avgPSActivity.mean(axis=1),
                            avgPSActivity.std(axis=1) / np.sqrt(avgPSActivity.shape[1]),
                            ps_T_corrected,
                            ax,
                            colorsPlt[i],
                            labels[i] if labels is not None else labels)


def compute_delay(tms, selectBlocksinfo, zeroMTCond, traceConds, activeNeu):
    delays = list()
    meanPSFR, t, meanBaselineFR, _ = tms.avg_FR_per_neuron(squeezeDim=False)
    _, zeroMTIdx = fb(selectBlocksinfo, zeroMTCond)
    for i in range(len(traceConds)):
        blocksInfo, blockIdx = fb(selectBlocksinfo, traceConds[i], ~zeroMTIdx)
        if blockIdx.sum() > 0:
            boolIndex = [activeNeu[item] if activeNeu[item].size > 1 else activeNeu[item][np.newaxis]
                         for item in blocksInfo.index]
            selectMeanPSFR = np.concatenate([meanPSFR[n][:, :, item][0, :, :]
                                             for n, item in zip(np.flatnonzero(blockIdx), boolIndex)],
                                            axis=1)
            d = tms.late_comp.compute_delay(selectMeanPSFR,
                                            t,
                                            selectMeanPSFR[t < 0, :].max(axis=0, keepdims=True),
                                            tms.analysis_params['peristimParams']['smoothingParams'][
                                                'width'] + 0.25)
            if np.isnan(d).sum() != len(d):
                delays.append(d[~np.isnan(d)])
            else:
                delays.append(np.array([np.nan, np.nan]))

        else:
            delays.append(np.array([np.nan, np.nan]))

    return delays


def plot_delay(delays, ax, colParams, colIdx, tms, selectBlocksinfo, zeroMTCond, traceConds, activeNeu,
               swarmplotsize=3):
    delays.update(
        {ascertain_colName_from_colParams(colParams[colIdx]):
             {key: delay for key, delay in zip([ascertain_colName_from_colParams(item) for item in traceConds],
                                               compute_delay(tms, selectBlocksinfo, zeroMTCond, traceConds,
                                                             activeNeu))}})
    sns.swarmplot(data=list(delays[ascertain_colName_from_colParams(colParams[colIdx])].values()),
                  color='k', size=swarmplotsize, ax=ax[1][colIdx])
    sns.violinplot(data=list(delays[ascertain_colName_from_colParams(colParams[colIdx])].values()),
                   inner=None, ax=ax[1][colIdx])

    pass