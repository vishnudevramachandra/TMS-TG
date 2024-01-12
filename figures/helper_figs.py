import numpy as np
from itertools import zip_longest
from tms_tg import EPOCHISOLATORS, FilterBlocks

fb = FilterBlocks()


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


def normalize_psfr(tms, fb, ps_T_corrected, ps_FR, blocksInfo):
    LATE_COMP_TIMEWIN = (5, 50)     # in msecs
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


def plot_MeanAndSEM(meanPSFR, semPSFR, t, ax, pltColor, lineLabel):
    ax.plot(t, meanPSFR, color=pltColor, label=lineLabel)
    ax.fill_between(t, meanPSFR - semPSFR, meanPSFR + semPSFR, alpha=0.2, color=pltColor)


def exclude_corrupt_traces(avgNormalizedPSFR):
    return np.delete(avgNormalizedPSFR, avgNormalizedPSFR.max(axis=0) > 20, axis=1)


def plot_populationAvgFR(tms, ps_T_corrected, ps_FR, selectBlocksinfo, traceConds, zeroMTCond, activeNeu, ax, rowIdx,
                         colIdx, colorsPlt, labels=None):
    normalizedPSFR = normalize_psfr(tms, fb, ps_T_corrected, ps_FR, selectBlocksinfo)
    _, zeroMTIdx = fb(selectBlocksinfo, zeroMTCond)
    if colIdx == 0:
        ax[rowIdx][colIdx].set_ylabel('Normalized\nfiring rate', fontsize=8)

    for i in range(len(traceConds)):
        _, blockIdx = fb(selectBlocksinfo, traceConds[i], ~zeroMTIdx)
        intIndices = np.where(blockIdx)[0]
        if len(intIndices) >= 1:
            avgNormalizedPSFR \
                = normalizedPSFR[intIndices[0]][:, :, activeNeu[blockIdx.index[intIndices[0]]]].mean(axis=0)
            for index in intIndices[1:]:
                boolIndex = activeNeu[blockIdx.index[index]]
                if boolIndex.size == 1:
                    boolIndex = boolIndex[np.newaxis]
                avgNormalizedPSFR = np.append(
                    avgNormalizedPSFR,
                    normalizedPSFR[index][:, :, boolIndex].mean(axis=0),
                    axis=1)

            avgNormalizedPSFR = exclude_corrupt_traces(avgNormalizedPSFR)
            plot_MeanAndSEM(avgNormalizedPSFR.mean(axis=1),
                            avgNormalizedPSFR.std(axis=1) / np.sqrt(avgNormalizedPSFR.shape[1]),
                            ps_T_corrected,
                            ax[rowIdx][colIdx],
                            colorsPlt[i],
                            labels[i] if labels is not None else labels)

