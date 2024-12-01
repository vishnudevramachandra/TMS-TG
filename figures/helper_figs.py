import numpy as np
from itertools import zip_longest
import pandas as pd
import copy
from functools import wraps
import matplotlib

from tms_tg import EPOCHISOLATORS, FilterBlocks

# Instantiate a FilterBlocks object
fb = FilterBlocks()


# Adjust the limits of the axes
def adjust_lim(ax, lim, kind):
    for axis in np.nditer(ax, flags=['refs_ok']):
        match kind:
            case 'xlim':
                axis.item().set_xlim(lim)
            case 'ylim':
                axis.item().set_ylim(lim)


# Generate column name from params argument
def colName_from_dict(colName, params):
    for key in params:
        if type(params[key]) == dict:
            colName = colName_from_dict(colName, params[key])
        elif params[key] is not None:
            colName += params[key] + '/'
    return colName


# Ascertain column name from column parameters
def ascertain_colName_from_colParams(colParams):
    if colParams is None:
        colName = 'All'
    else:
        colName = colName_from_dict('', colParams)
    return colName


# Normalize peristimulus firing rate
def normalize_psfr(ps_T_corrected, ps_FR, blocksInfo):
    LATE_COMP_TIMEWIN = (5, 50)  # in msecs
    PARAMS = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}}
    _, mtBlocksIdx = fb(blocksInfo, PARAMS)
    normalizedPSFR = [None] * len(ps_FR)

    for epochIndex in blocksInfo.index.unique():
        boolEpochIndex = blocksInfo.index == epochIndex

        # Take PSFR at motor threshold
        mtIndices = np.where(boolEpochIndex & mtBlocksIdx.to_numpy())[0]
        mtPSFR = ps_FR[mtIndices[0]]
        for index in mtIndices[1:]:
            mtPSFR = np.append(mtPSFR, ps_FR[index], axis=0)
        avg_mtPSFR = mtPSFR.mean(axis=0, keepdims=True)
        peakFR_latecomp = \
            (avg_mtPSFR[:, (LATE_COMP_TIMEWIN[0] <= ps_T_corrected) & (ps_T_corrected < LATE_COMP_TIMEWIN[1]), :].
             max(axis=1, keepdims=True))

        # Take the peak firing rate of the late component for PSFR at motor threshold and do the normalization
        for i in np.where(boolEpochIndex)[0]:
            normalizedPSFR[i] = ps_FR[i] / peakFR_latecomp

    return normalizedPSFR


# Randomly select a neuron
def selectNeuron(activeNeu):
    return np.random.choice(activeNeu)


# Randomly select an epoch and a neuron for plotting
def selectEpochAndNeuron(epochAndNeuron, tms, epochIndices, activeNeus, colParam):
    # Randomly sample an epoch that has at least one active neuron for plotting if the choice of Epoch is unspecified,
    # i.e., if EpochAndNeuron is None
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

    # Randomly select a neuron for plotting raster if choice of Neuron is unspecified, i.e., if EpochAndNeuron is None
    neuIdx = np.random.choice(sampleActiveNeus.nonzero()[0]) \
        if epochAndNeuron is None or 'neuIdx' not in epochAndNeuron.keys() else epochAndNeuron['neuIdx']

    sampleBlocksinfo, _ = tms.filter_blocks

    return sampleBlocksinfo, sampleEpochIndex, neuIdx


# Plot mean and standard error of peristimulus firing rate
def plot_MeanAndSEM(meanPSFR, semPSFR, t, ax, pltColor, lineLabel):
    ax.plot(t, meanPSFR, color=pltColor, label=lineLabel)
    ax.fill_between(t, meanPSFR - semPSFR, meanPSFR + semPSFR, alpha=0.2, color=pltColor)


# Exclude corrupt traces from average population activity
def exclude_corrupt_traces(avgPSActivity):
    return np.delete(avgPSActivity, avgPSActivity.max(axis=0) > 20, axis=1)


# Compute exclude index based on selection conditions
def compute_excludeIdx(selectBlocksinfo, excludeConds):
    excludeIdx = pd.Series(False, index=selectBlocksinfo.index)
    if excludeConds is not None:
        for excludeCond in excludeConds:
            excludeIdx |= fb(selectBlocksinfo, excludeCond)[1]
    return excludeIdx


# Select activity based on exclude index
def select_activity(psActivity, selectBlocksinfo, selectBlocksinfoIdx, activeNeus, excludeIdx):
    selectBlocksinfo = selectBlocksinfo.loc[~excludeIdx]
    return ([psActivity[i] for i in np.nonzero(~excludeIdx)[0]],
            selectBlocksinfo,
            list(map(lambda x: activeNeus[x] if activeNeus[x].size > 1 else activeNeus[x][np.newaxis],
                     selectBlocksinfo.index)),
            np.nonzero(selectBlocksinfoIdx)[0][~excludeIdx])


# Plot population average firing rate
def plot_populationAvgFR(tms, activeNeus, ax, kind='Average', excludeConds=None, lineLabel=None, lineColor=None):
    psActivity, t, _, _ = tms.avg_FR_per_neuron(squeezeDim=False)
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
    excludeIdx = compute_excludeIdx(selectBlocksinfo, excludeConds)

    if all(excludeIdx):
        return None

    if kind == 'Normalized':
        psActivity, doneIdx = normalize_psfr(tms, psActivity, t)
        excludeIdx |= ~doneIdx

    # Select items which are not marked for exclusion
    selectpsActivity, selectBlocksinfo, selectActNeus, _ \
        = select_activity(psActivity, selectBlocksinfo, selectBlocksinfoIdx, activeNeus, excludeIdx)

    # avgPSActivity = exclude_corrupt_traces(avgPSActivity)
    plot_MeanAndSEM(np.concatenate(selectpsActivity, axis=2)[0][:, np.concatenate(selectActNeus)].mean(axis=1),
                    np.concatenate(selectpsActivity, axis=2)[0][:, np.concatenate(selectActNeus)].std(axis=1)
                    / np.sqrt(sum(np.concatenate(selectActNeus))),
                    t,
                    ax,
                    lineColor,
                    lineLabel)


# Retrieve or compute-and-retrieve delay based on peristimulus activity
def pick_delay(tms, activeNeus, excludeConds=None):
    meanPSFR, t, _, _ = tms.avg_FR_per_neuron(squeezeDim=False)
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
    excludeIdx = compute_excludeIdx(selectBlocksinfo, excludeConds)

    if all(excludeIdx):
        return None

    # Select items which are not marked for exclusion
    selectMeanPSFR, selectBlocksinfo, selectActNeus, selectIndices \
        = select_activity(meanPSFR, selectBlocksinfo, selectBlocksinfoIdx, activeNeus, excludeIdx)

    # Use pre-computed delay if it exists otherwise compute it
    if 'delay' not in tms.blocksinfo.columns:
        tms.blocksinfo['delay'] = None
        tms.filter_blocks = None
    delayColIdx = np.nonzero(tms.blocksinfo.columns == 'delay')[0][0]
    d = list()
    for idx, fIdx in enumerate(selectIndices):
        if (tms.blocksinfo.iat[fIdx, delayColIdx] is not None
                and tms.late_comp.delayMethod in (cell := tms.blocksinfo.iat[fIdx, delayColIdx]).keys()):
            d.extend(cell[tms.late_comp.delayMethod])
            continue

        tmp_d = tms.late_comp.compute_delay(selectMeanPSFR[idx][0, :, :],
                                            t,
                                            selectMeanPSFR[idx][0, t < 0, :].max(axis=0, keepdims=True),
                                            tms.analysis_params['peristimParams']['smoothingParams']['width'] + 0.25)
        if (cell := tms.blocksinfo.iat[fIdx, delayColIdx]) is not None:
            cell[tms.late_comp.delayMethod] = tmp_d
        else:
            tms.blocksinfo.iat[fIdx, delayColIdx] = {tms.late_comp.delayMethod: tmp_d}

        tms.filter_blocks = None
        d.extend(tmp_d)

    return np.array(d)[np.concatenate(selectActNeus)]


# Add additional information to grand blocksinfo
def gb_addinfo(gbinfo, tms):
    """

    Parameters
    ----------
    gbinfo      : [Dataframe] 'grand' blocksinfo from all animals
    tms         : [Dataframe] subset of blocksinfo containing extra data as a result of some computation

    Returns
    -------
    [Dataframe] 'grand' blocksinfo, now with added data
    """
    bIndicesInGBIndices = tuple(set(tms.blocksinfo.index.unique()) & set(gbinfo.index.unique()))
    for index in bIndicesInGBIndices:
        assert all(gbinfo.loc[index, ['MSO ', 'MT']] == tms.blocksinfo.loc[index, ['MSO ', 'MT']]), \
            'the row order of grandBlocksinfo is dissimilar to that of tms-blocksinfo'
        gbinfo.loc[index, 'psActivity'] = \
            (pd.concat([gbinfo.loc[index, 'psActivity'].rename('x'),
                        tms.blocksinfo.loc[index, 'psActivity'].rename('y')], axis=1).
             apply(lambda s: (s.x | s.y if isinstance(s.y, dict) else s.x) if isinstance(s.x, dict) else s.y,
                   axis=1))
        gbinfo.loc[index, 'delay'] = \
            (pd.concat([gbinfo.loc[index, 'delay'].rename('x'),
                        tms.blocksinfo.loc[index, 'delay'].rename('y')], axis=1).
             apply(lambda s: (s.x | s.y if isinstance(s.y, dict) else s.x) if isinstance(s.x, dict) else s.y,
                   axis=1))

    bIndicesMinusGBIndices = tuple(set(tms.blocksinfo.index.unique()) - set(gbinfo.index.unique()))
    for index in bIndicesMinusGBIndices:
        gbinfo = pd.concat([gbinfo, tms.blocksinfo.loc[index, :]], axis=0)

    return gbinfo.sort_index()


# From a collection of keys find the closest matching key to an exemplar key
def closest_matching_key(keys, inputKey):
    df = pd.DataFrame([
        [key,
         [0 if x == y else np.inf if isinstance(x, str) else abs(x - y) for x, y in zip(key, inputKey)]
         ]
        for key in keys
    ])
    return df.iat[df.iloc[:, 1].apply(lambda cell: np.nansum(cell)).argmin(), 0]


# Read peristimulus activity from blocksinfo dataframe
def read_PSFR(psActivity, key):
    matchKey = closest_matching_key(psActivity.keys(), key)
    return psActivity[matchKey]


# Decorator for aggregation functions
def agg_dec(fcn):
    @wraps(fcn)
    def arg_filter(arg, **kwargs):
        match fcn.__name__:
            case 'delay_agg':
                return fcn(arg, **kwargs)
            case 'pkFr_agg':
                return fcn(arg, bIdx=kwargs['bIdx'])

    return arg_filter


# Aggregate computed delay
@agg_dec
def delay_agg(psActivity, bIdx, fcn, peakWidth):
    ps_FR, ps_T = read_PSFR(psActivity, ('gaussian', np.nan, np.nan, np.nan, 50.0, 'TMS'))
    return fcn(ps_FR.mean(axis=0),
               ps_T,
               ps_FR.mean(axis=0)[ps_T < 0, :].max(axis=0, keepdims=True),
               peakWidth)[bIdx]


# Aggregate computed peak firing rates
@agg_dec
def pkFr_agg(psActivity, bIdx):
    psFR, t = read_PSFR(psActivity, ('gaussian', np.nan, np.nan, np.nan, 50.0, 'TMS'))
    return np.concatenate(psFR, axis=0).mean(axis=0)[np.ix_(np.logical_and(t > 5.0, t <= 50.0), bIdx)].max(axis=0)


# Rotate a list
def rotate(array: list, step: int):
    return array[-step:] + array[:-step]


# Extract grand data from sub-frames
def gp_extractor(subf, aggFcn, activeNeus, silencingType, extrcCol, postTi, tms):
    epochIdx = subf.index.unique()
    match silencingType:
        case 'TG-Injection ' | 'Skin-Injection' | 'TGOrifice':
            postT = subf[silencingType].str.extract('(\d+)').astype('float64')[0]
            gpOut = pd.DataFrame()
            mtTypes = subf['MT'].unique()
            for mtType in mtTypes:
                selectMTIdx = subf['MT'] == mtType
                out = [subf.loc[blockIdx, extrcCol].agg(aggFcn,
                                                        axis=0,
                                                        bIdx=activeNeus[epochIdx].item(),
                                                        fcn=tms.late_comp.compute_delay,
                                                        peakWidth=
                                                        tms.analysis_params['peristimParams']['smoothingParams'][
                                                            'width'] + 0.25)
                       if any(blockIdx := subf[silencingType].str.contains('Pre') & selectMTIdx)
                       else np.repeat(np.nan, activeNeus[epochIdx].item().sum())]

                for ti in postTi:
                    blockIdx = (ti[0] <= postT) & (postT < ti[1]) & selectMTIdx
                    if any(blockIdx):
                        out.append(subf.loc[blockIdx, extrcCol].agg(aggFcn,
                                                                    axis=0,
                                                                    bIdx=activeNeus[epochIdx].item(),
                                                                    fcn=tms.late_comp.compute_delay,
                                                                    peakWidth=(tms.analysis_params['peristimParams']
                                                                               ['smoothingParams']['width'] + 0.25)))
                    else:
                        out.append(np.repeat(np.nan, activeNeus[epochIdx].item().sum()))

                gpOut = pd.concat((gpOut,
                                   pd.DataFrame({key: value for key, value in zip(['Pre', *map(str, postTi), 'MT'],
                                                                                  [*out, mtType])})))

            return gpOut
        case 'TGcut':
            return (subf[extrcCol]
                    .apply(aggFcn,
                           bIdx=activeNeus[epochIdx].item(),
                           fcn=tms.late_comp.compute_delay,
                           peakWidth=(tms.analysis_params['peristimParams']
                                      ['smoothingParams']['width'] + 0.25))
                    .apply(pd.Series)
                    .set_index(pd.MultiIndex.from_product([['MT'], subf.loc[:, 'MT'].to_numpy()]))
                    .T)


# Modify violin plot to have unfilled areas
def violin_fill_false(ax):
    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_edgecolor(collection.get_facecolor())
            collection.set_facecolor('none')
    for h in ax.legend_.legend_handles:
        if isinstance(h, matplotlib.patches.Rectangle):
            h.set_edgecolor(h.get_facecolor())
            h.set_facecolor('none')
            h.set_linewidth(1.5)
