import itertools

import numpy as np
import pandas as pd
import re
from functools import lru_cache, wraps
from lib.constants import COLS_WITH_STRINGS
import scipy
import warnings
from typing import Optional, Any


class AnalysisParams(object):
    """
    Data descriptor that stores parameters which are used for data selection and data analysis
    """

    def __init__(self, params):
        self.__set__(self, params)

    def __set__(self, obj: 'TMSTG', params):
        if self == obj:
            self.analysis_params = params

        else:
            try:
                if not isinstance(params, dict):
                    params = self.analysis_params
                    raise ValueError

                raiseFlag = False

                if 'selectionParams' in params.keys():
                    if 'Epoch' not in params['selectionParams'].keys() \
                            or \
                            not issubclass(type(params['selectionParams']['Epoch']), dict) \
                            or \
                            not (params['selectionParams']['Epoch'].keys()
                                 & self.analysis_params['selectionParams']['Epoch'].keys()):
                        params['selectionParams']['Epoch'] \
                            = self.analysis_params['selectionParams']['Epoch']
                        raiseFlag = True
                    else:
                        for epochKey in self.analysis_params['selectionParams']['Epoch']:
                            if epochKey in params['selectionParams']['Epoch'].keys():
                                if not issubclass(type(params['selectionParams']['Epoch'][epochKey]),
                                                  tuple | set | list | np.ndarray | None):
                                    params['selectionParams']['Epoch'][epochKey] \
                                        = (params['selectionParams']['Epoch'][epochKey],)
                            else:
                                params['selectionParams']['Epoch'][epochKey] \
                                    = self.analysis_params['selectionParams']['Epoch'][epochKey]

                    for col in params['selectionParams'].keys() & COLS_WITH_STRINGS:
                        if not issubclass(type(params['selectionParams'][col]), tuple | set | list | np.ndarray):
                            params['selectionParams'][col] = (params['selectionParams'][col],)

                else:
                    params['selectionParams'] = self.analysis_params['selectionParams']

                if 'TMSArtifactParams' in params.keys() and 'timeWin' in params['TMSArtifactParams'].keys():
                    if len(params['TMSArtifactParams']['timeWin']) == 2:
                        params['TMSArtifactParams']['timeWin'] = tuple(params['TMSArtifactParams']['timeWin'])
                        obj.singleUnitsSpikeTimes.cache_clear()
                    else:
                        params['TMSArtifactParams']['timeWin'] \
                            = self.analysis_params['TMSArtifactParams']['timeWin']
                        raiseFlag = True
                else:
                    params['TMSArtifactParams'] = self.analysis_params['TMSArtifactParams']

                if 'peristimParams' in params.keys():
                    if 'smoothingParams' in params['peristimParams'].keys():
                        unentered_keys = self.analysis_params['peristimParams']['smoothingParams'].keys() \
                                         - params['peristimParams']['smoothingParams'].keys()
                        for key in unentered_keys:
                            params['peristimParams']['smoothingParams'][key] \
                                = self.analysis_params['peristimParams']['smoothingParams'][key]
                        if unentered_keys == self.analysis_params['peristimParams']['smoothingParams'].keys():
                            raiseFlag = True
                    else:
                        params['peristimParams']['smoothingParams'] \
                            = self.analysis_params['peristimParams']['smoothingParams']

                    if 'timeWin' in params['peristimParams'].keys():
                        if len(params['peristimParams']['timeWin']) == 2:
                            params['peristimParams']['timeWin'] = tuple(params['peristimParams']['timeWin'])
                        else:
                            params['peristimParams']['timeWin'] = self.analysis_params['peristimParams']['timeWin']
                            raiseFlag = True
                    else:
                        params['peristimParams']['timeWin'] = self.analysis_params['peristimParams']['timeWin']

                    if 'trigger' in params['peristimParams'].keys():
                        # TODO: random trigger implementation
                        pass
                    else:
                        params['peristimParams']['trigger'] = self.analysis_params['peristimParams']['trigger']

                    if 'baselinetimeWin' in params['peristimParams'].keys():
                        pass
                    else:
                        params['peristimParams']['baselinetimeWin'] \
                            = self.analysis_params['peristimParams']['baselinetimeWin']
                else:
                    params['peristimParams'] = self.analysis_params['peristimParams']

                if raiseFlag:
                    raise ValueError

            except ValueError:
                print(f'analysis_params does not adhere to correct format, '
                      f'using instead default/previous params...')

            print('analysis_params set to: ', params)
            print('----------------------------')
            self.analysis_params = params

            # do housekeeping
            if 'psts' in obj.__dict__:
                del obj.psts
            obj.compute_firingrate.cache_clear()
            obj.filter_blocks = None
            _, _ = obj.filter_blocks

    def __get__(self, obj, objType):
        return self.analysis_params


def _get_trigger_times_for_current_block(trigType, matdatum, blockinfo):
    if trigType == 'TMS':
        trigger = _read_trigger(matdatum)
        return trigger[blockinfo['TrigStartIdx'] + np.array(range(blockinfo['no. of Trigs']))]
    else:
        ...  # TODO: random trigger implementation


@lru_cache(maxsize=None)
def _read_trigger(matdatum):
    trigChanIdx = matdatum['TrigChan_ind'][0, 0].astype(int) - 1
    refs = matdatum['rawData/trigger'].flatten()
    return matdatum[refs[trigChanIdx]].flatten()[::2] * 1e3


def _check_trigger_numbers(matdata, blocksinfo) -> None:
    epochIndices = blocksinfo.index.unique().to_numpy()
    for epochIndex in epochIndices:
        trigger = _read_trigger(matdata[epochIndex])
        assert blocksinfo.loc[epochIndex, 'no. of Trigs'].sum() == len(trigger), \
            f'no. of triggers in epoch {epochIndex} does not match with mat-data ({len(trigger)})'
        # matdata[epochIndex][matdata[epochIndex]['CombiMCD_fnames'].flatten()[0]].tobytes().decode('utf-16')


@lru_cache(maxsize=None)
def _read_MSO(matdatum) -> np.ndarray:
    refs = matdatum['blockInfo/MSO'].flatten()
    mso = [matdatum[i].flatten().tobytes().decode('utf-16') for i in refs]
    return np.array(
        [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else 0 for item in mso])


def _rowExpander(s):
    ret = list()
    indices = s.index
    [indices := indices[indices != item] for item in ['MSO_order', 'TrigIndices']]
    for idx in indices:
        match idx:
            case 'MSO ' | 'no. of Trigs' | 'Stimpulses':
                ret.append(s['MSO_order'][idx].to_list())
            case 'MT' | 'StimHem' | 'CoilDir' | 'TG-Injection ' | 'RecArea ' | 'RecHem' | 'Depth_int' | 'Movement':
                ret.append([s[idx]] * s['MSO_order'].shape[0])
            case 'Filename':
                ret.append(s[idx].split(','))
            case 'Queries ' | 'Skin-Injection' | 'Comments' | 'Time':
                if isinstance(s[idx], list):
                    ret.append(s[idx])
                else:
                    ret.append([s[idx]] * s['MSO_order'].shape[0])

    return pd.DataFrame([[ret[j][i] for j in range(len(ret))] for i in range(len(ret[0]))], columns=indices)


def _check_mso_order(matdata, blocksinfo) -> None:
    epochIndices = blocksinfo.index.unique().to_numpy()
    for epochIndex in epochIndices:
        if any([re.search(r'/', item) for item in
                [matdata[epochIndex][ref].flatten().tobytes().decode('utf-16')
                 for ref in matdata[epochIndex]['blockInfo/MSO'].flatten()]]):
            warnings.warn(f'Cannot check the MSO order in blocksinfo for epoch {epochIndex} using mat-data information.'
                          f' Hence, using the MSO order in blocksinfo as it is; keep in mind that the source of this '
                          f'MSO order contained in infofile has to match the true recording MSO order, otherwise wrong '
                          f'MSO will be associated to epochs [Katastrophe].')
            continue

        mso = _read_MSO(matdata[epochIndex])
        infofileMSOorder = np.concatenate(
            blocksinfo.loc[epochIndex, 'MSO_order'].apply(lambda x: x[['MSO_order', 'MSO ']].to_numpy()))
        argsort = np.argsort(infofileMSOorder[:, 0])
        infofileFilenames = np.concatenate(
            blocksinfo.loc[epochIndex, 'Filename'].apply(lambda x: x.split(',')).to_numpy())[argsort]
        infofileMSOorder = infofileMSOorder[argsort, 1]
        nonZeroMSOindices = infofileMSOorder != 0
        try:
            assert all(infofileMSOorder[nonZeroMSOindices] == mso[nonZeroMSOindices])

        except AssertionError:
            # check the case where filenames were not correctly labeled with _mso values
            refs = matdata[epochIndex]['CombiMCD_fnames'].flatten()
            combiMCDFnames = pd.Series([matdata[epochIndex][i].flatten().tobytes().decode('utf-16') for i in refs])
            fnames = combiMCDFnames.str.rsplit(pat='\\', n=1, expand=True)[1]
            if not all(fnames.values == infofileFilenames):
                warnings.warn(f'MSO order in blocksinfo for epoch {epochIndex} differs from mat-data.'
                              f'Hence, the order in blocksinfo is being changed to match mat-data')
                index = list()
                for row in infofileFilenames:
                    index.append(np.argwhere(fnames.str.fullmatch(row)).item())
                assert len(index) == len(infofileFilenames), \
                    f'The filenames in blocksinfo for epoch {epochIndex} do not match combiMCDFnames'
                df = blocksinfo.loc[epochIndex, :].copy().reset_index(drop=True)
                df = df.apply(_rowExpander, axis=1, result_type='expand')
                [row for idx, row in df.iterrows()]
                blocksinfo.loc[epochIndex, :].iloc[index, :] = tmp
                blocksinfo.drop(columns='TrigIndices', inplace=True)
                _concat_blocksinfo(blocksinfo, 'TrigIndices')


def _concat_blocksinfo(blocksinfo: pd.DataFrame, colName: str, value: Optional[Any] = None) -> None:
    """
    Adds a new column to passed DataFrame

    Parameters
    ----------
    blocksinfo:  [DataFrame] to which a new column is concatenated
    colName:     name for the newly added column
    value:       [Optional] all rows of the newly added column are set to this value

    Returns
    -------
    None
    """

    if colName not in blocksinfo.columns:
        blocksinfo[colName] = value


def _rowCombiner(subf, cols):
    ret = list()
    for col in cols:
        match col:
            case 'MSO ' | 'Depth_int':
                ret.append(subf[col].mean().astype('int'))
            case 'no. of Trigs' | 'Stimpulses':
                ret.append(subf[col].sum())
            case 'TrigIndices':
                ret.append(np.concatenate(subf[col]))
            case 'Filename':
                ret.append(','.join(subf[col]))
            case 'MSO_order':
                ret.append(subf[[col, 'MSO ', 'no. of Trigs', 'Stimpulses']].reset_index(drop=True))

    return pd.Series(ret, index=cols)


def _edit_blocksinfo(blocksinfo: pd.DataFrame, cond: str) -> pd.DataFrame:
    """
    Edits blocksinfo so that each row pertains to a unique experimental condition

    Parameters
    ----------
    blocksinfo:     MultiIndex-ed [DataFrame] containing block information
    cond:           new info added to dataframe

    Returns
    -------
    edited DataFrame
    """

    if cond not in ('TrigIndices',):
        raise NotImplementedError(f'editing blocksinfo while using {cond} as argument is not implemented')

    blocksinfo['MSO_order'] = np.array(range(blocksinfo.shape[0]))
    for epochIndex in blocksinfo.index.unique():
        num_of_trigs = blocksinfo.loc[epochIndex, 'no. of Trigs'].to_numpy()
        stimpulses = blocksinfo.loc[epochIndex, 'Stimpulses'].to_numpy()
        blocksinfo.loc[epochIndex, 'TrigIndices'] = (
                pd.Series([np.array(range(x)) for x in
                           [b if (b <= a * 0.75) else a for a, b in zip(num_of_trigs, stimpulses)]])
                + np.append(0, num_of_trigs.cumsum()[:-1])).values

    gpExcludeCols = ['MSO ', 'no. of Trigs', 'Stimpulses', 'Filename', 'TrigIndices', 'Depth_int', 'MSO_order']
    gp = blocksinfo.groupby(blocksinfo.index.names + list(np.setdiff1d(blocksinfo.columns, gpExcludeCols)),
                            group_keys=True,
                            sort=False)
    return (gp.apply(lambda x: _rowCombiner(x, gpExcludeCols)).
            reset_index(level=list(np.setdiff1d(blocksinfo.columns, gpExcludeCols)))
            [blocksinfo.columns])


class LateComponent(object):
    """
    Class dealing with analysis of late activity
    """
    _methodsForComputingDelay = {'threshold_crossing', 'starting_point_derived_from_slope', 'peak_as_delay'}
    _defaultMethodForComputingDelay = 'starting_point_derived_from_slope'
    earlyLateSeparationTimePoint = 5  # in msecs

    def __init__(self, method=None):
        self.delayMethod = LateComponent._defaultMethodForComputingDelay if method is None else method

    @property
    def delayMethod(self):
        return self._delayMethod

    @delayMethod.setter
    def delayMethod(self, method):
        if method in self._methodsForComputingDelay:
            self._delayMethod = method
        elif hasattr(self, '_delayMethod'):
            print('wrong method passed for computing delay, reverting to previous method')
        else:
            self._delayMethod = LateComponent._defaultMethodForComputingDelay
            print('wrong method passed during initiation, using default condition for computing delay')
        print(f'Method for computing delay set to: {self._delayMethod}')

    def compute_delay(self, meanPSFR, ps_T, meanBaselineFR, minPeakWidth) -> np.ndarray:
        """
        Compute the delay of activity with respect to trigger using one the chosen 'delayMethod'

        Parameters
        ----------
        meanPSFR        :  average peristimulus activity; array[Time X N]
        ps_T            :  Time points of peristimulus activity; array[1D]
        meanBaselineFR  :  baseline firing rate; array[1 X N]
        minPeakWidth    :  (ms); the cutoff criteria for selecting the peaks

        Returns
        -------
        array[1D] of delays (size N)
        """
        match self.delayMethod:
            case 'baseline_crossing':
                fcn = self.threshold_crossing
            case 'starting_point_derived_from_slope':
                fcn = self.starting_point_derived_from_slope
            case 'peak_as_delay':
                fcn = self.peak_as_delay

        dt = ps_T[1] - ps_T[0]
        earlyLateSeparationIdx = (self.earlyLateSeparationTimePoint - ps_T[0]) / dt
        delays = np.empty(shape=0, dtype=float)
        for colIdx in range(meanPSFR.shape[1]):
            # although the value 10 may seem arbitrary, it is appropriate for this dataset
            minPeakHeight = 2 * meanBaselineFR[0, colIdx] if meanBaselineFR[0, colIdx] > 0.1 else 10
            peaks, _ = scipy.signal.find_peaks(meanPSFR[:, colIdx], height=minPeakHeight, width=minPeakWidth / dt)
            latePeaksIdx = np.argwhere(peaks >= earlyLateSeparationIdx)
            if len(latePeaksIdx) > 0:
                delays = np.append(delays, fcn(peaks.item(latePeaksIdx.item(0)),
                                               dt,
                                               ps_T[0],
                                               meanPSFR[:, colIdx],
                                               meanBaselineFR[0, colIdx],
                                               minPeakHeight,
                                               minPeakWidth,
                                               earlyLateSeparationIdx))
            else:
                delays = np.append(delays, np.nan)
        return delays

    @staticmethod
    def delay_deco(fcn):
        @wraps(fcn)
        def arg_filter(*args):
            match fcn.__name__:
                case 'baseline_crossing':
                    return fcn(*args[:5])
                case 'starting_point_derived_from_slope':
                    return fcn(*args)
                case 'peak_as_delay':
                    return fcn(*args[:3])

        return arg_filter

    @staticmethod
    @delay_deco
    def threshold_crossing(peakIdx, dt, offset, waveform, baselineFR):
        if baselineFR > 0:
            return np.max(np.argwhere(waveform[:peakIdx] <= 1.5 * baselineFR), initial=0) * dt + offset
        assert offset < 0, ('computed peristimulus firing does not have pre-stimulus activity, '
                            'hence cannot compute baseline_crossing for zero-valued baselineFR')
        baselineFR = waveform[:int(dt * -offset)].mean()
        return np.max(np.argwhere(waveform[:peakIdx] <= 1.5 * baselineFR), initial=0) * dt + offset

    @staticmethod
    @delay_deco
    def starting_point_derived_from_slope(
            peakIdx, dt, offset, waveform, baselineFR, minPeakHeight, minPeakWidth, earlyLateSeparationIdx):
        df1 = np.diff(waveform, n=1)
        peaks, _ = scipy.signal.find_peaks(df1)
        shadowPeak = find_shadowPeak(peakIdx, dt, waveform, minPeakWidth, scipy.signal.find_peaks(-df1)[0])
        if shadowPeak >= earlyLateSeparationIdx and waveform[shadowPeak] > minPeakHeight:
            peakIdx = shadowPeak
        inflectionIdx = peaks[np.argwhere(peaks < peakIdx)].max()
        return (offset
                + (inflectionIdx + 0.5 -
                   ((waveform[inflectionIdx:inflectionIdx + 2].mean() - baselineFR) / df1[inflectionIdx])) * dt)

    @staticmethod
    @delay_deco
    def peak_as_delay(peakIdx, dt, offset):
        return (peakIdx * dt) + offset


def find_shadowPeak(peakIdx, dt, waveform, minPeakWidth, troughs):
    troughs = troughs[troughs < peakIdx]
    if troughs.size > 0:
        return troughs[-1] + 1 - waveform[troughs[-1] - range(-1, np.rint(minPeakWidth / dt).astype(int))].argmax()
    return np.nan



