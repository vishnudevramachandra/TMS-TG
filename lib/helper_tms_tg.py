import numpy as np
import pandas as pd
import re
from functools import lru_cache, wraps, reduce
from lib.constants import COLS_WITH_STRINGS
import scipy
import warnings
from typing import Optional, Any


class AnalysisParams(object):
    """AnalysisParams

    Data descriptor for managing parameters that are used for data selection and data analysis

    """

    def __init__(self, params):
        # Initialize the AnalysisParams object with parameters
        self.__set__(self, params)

    def __set__(self, obj: 'TMSTG', params):
        # Check if the current instance is being set during 'TMSTG' instantiation
        if self == obj:
            # If yes, set the analysis_params attribute to default parameters
            self.analysis_params = params

        else:
            try:
                # Check if params is not a dictionary
                if not isinstance(params, dict):
                    # If not, use the previously set analysis_params
                    params = self.analysis_params
                    # Raise a ValueError
                    raise ValueError

                # Initialize a flag to track if any of the input parameters does not adhere to prescribed format
                raiseFlag = False

                # Check if selectionParams key exists in params dictionary
                if 'selectionParams' in params.keys():
                    # Check if 'Epoch' key exists in selectionParams
                    if 'Epoch' not in params['selectionParams'].keys() \
                            or \
                            not issubclass(type(params['selectionParams']['Epoch']), dict) \
                            or \
                            not (params['selectionParams']['Epoch'].keys()
                                 & self.analysis_params['selectionParams']['Epoch'].keys()):
                        # If any of the conditions fail, use the default Epoch parameters
                        params['selectionParams']['Epoch'] \
                            = self.analysis_params['selectionParams']['Epoch']
                        # Set the raiseFlag to True indicating input parameters did not adhere to prescribed format
                        raiseFlag = True
                    else:
                        # Iterate over each epoch key
                        for epochKey in self.analysis_params['selectionParams']['Epoch']:
                            # Check if the key exists in params
                            if epochKey in params['selectionParams']['Epoch'].keys():
                                # Check if the value is not a tuple, set, list, ndarray, or None
                                if not issubclass(type(params['selectionParams']['Epoch'][epochKey]),
                                                  tuple | set | list | np.ndarray | None):
                                    # If not, convert it to a tuple
                                    params['selectionParams']['Epoch'][epochKey] \
                                        = (params['selectionParams']['Epoch'][epochKey],)
                            else:
                                # If key doesn't exist, use previous value
                                params['selectionParams']['Epoch'][epochKey] \
                                    = self.analysis_params['selectionParams']['Epoch'][epochKey]

                else:
                    # If selectionParams key doesn't exist, use previous value
                    params['selectionParams'] = self.analysis_params['selectionParams']

                # Check if TMSArtifactParams key exists in params dictionary and 'timeWin' key exists in it
                if 'TMSArtifactParams' in params.keys() and 'timeWin' in params['TMSArtifactParams'].keys():
                    # Check if the length of timeWin is 2
                    if len(params['TMSArtifactParams']['timeWin']) == 2:
                        # If yes, convert it to a tuple
                        params['TMSArtifactParams']['timeWin'] = tuple(params['TMSArtifactParams']['timeWin'])
                        # Clear cache for singleUnitsSpikeTimes
                        obj.singleUnitsSpikeTimes.cache_clear()
                    else:
                        # If not, use the previous value
                        params['TMSArtifactParams']['timeWin'] \
                            = self.analysis_params['TMSArtifactParams']['timeWin']
                        # Set the raiseFlag to True indicating input parameters did not adhere to prescribed format
                        raiseFlag = True
                else:
                    # If key doesn't exist, use previous value
                    params['TMSArtifactParams'] = self.analysis_params['TMSArtifactParams']

                # Check if peristimParams key exists in params dictionary
                if 'peristimParams' in params.keys():
                    # Check if 'smoothingParams' key exists in peristimParams
                    if 'smoothingParams' in params['peristimParams'].keys():
                        # Get the unentered keys
                        unentered_keys = self.analysis_params['peristimParams']['smoothingParams'].keys() \
                                         - params['peristimParams']['smoothingParams'].keys()

                        # Iterate over unentered keys
                        for key in unentered_keys:
                            # Set unentered keys to default values
                            params['peristimParams']['smoothingParams'][key] \
                                = self.analysis_params['peristimParams']['smoothingParams'][key]
                        # Check if all smoothingParams keys are unentered
                        if unentered_keys == self.analysis_params['peristimParams']['smoothingParams'].keys():
                            # Set the raiseFlag to True indicating input parameters did not adhere to prescribed format
                            raiseFlag = True
                    else:
                        # If key doesn't exist, use previous value
                        params['peristimParams']['smoothingParams'] \
                            = self.analysis_params['peristimParams']['smoothingParams']

                    # Check if 'timeWin' key exists in peristimParams
                    if 'timeWin' in params['peristimParams'].keys():
                        # Check if the length of timeWin is 2
                        if len(params['peristimParams']['timeWin']) == 2:
                            # If yes, convert it to a tuple
                            params['peristimParams']['timeWin'] = tuple(params['peristimParams']['timeWin'])
                        else:
                            # If not, use the previous value
                            params['peristimParams']['timeWin'] = self.analysis_params['peristimParams']['timeWin']
                            # Set the raiseFlag to True indicating input parameters did not adhere to prescribed format
                            raiseFlag = True
                    else:
                        # If key doesn't exist, use previous value
                        params['peristimParams']['timeWin'] = self.analysis_params['peristimParams']['timeWin']

                    # Check if 'trigger' key exists in peristimParams
                    if 'trigger' in params['peristimParams'].keys():
                        # TODO: random trigger implementation
                        raise PermissionError('trigger cannot be set for analysis_params')
                    else:
                        # If key doesn't exist, use previous value
                        params['peristimParams']['trigger'] = self.analysis_params['peristimParams']['trigger']

                    # Check if 'baselinetimeWin' key exists in peristimParams
                    if 'baselinetimeWin' in params['peristimParams'].keys():
                        # Raise a PermissionError as 'baselinetimeWin' cannot be set for analysis_params
                        raise PermissionError('baselinetimeWin cannot be set for analysis_params')
                    else:
                        # If key doesn't exist, use previous value
                        params['peristimParams']['baselinetimeWin'] \
                            = self.analysis_params['peristimParams']['baselinetimeWin']
                else:
                    # If key doesn't exist, use previous value
                    params['peristimParams'] = self.analysis_params['peristimParams']

                # Check if raiseFlag is True
                if raiseFlag:
                    # If yes, raise a ValueError
                    raise ValueError

            except ValueError:
                # Catch the ValueError and print a message
                print(f'analysis_params does not adhere to correct format, '
                      f'using instead default/previous params...')

            # Print the set parameters
            print('analysis_params set to: ', params)
            print('----------------------------')
            # Set the analysis_params attribute to the updated params
            self.analysis_params = params

            # do housekeeping
            if 'psts' in obj.__dict__:
                del obj.psts
            obj.compute_firingrate.cache_clear()
            obj.filter_blocks = None
            _, _ = obj.filter_blocks

    def __get__(self, obj, objType):
        # Return the analysis parameters
        return self.analysis_params


# Function to get trigger times for the current block
def _get_trigger_times_for_current_block(trigType, matdatum, trigIndices):
    if trigType == 'TMS':
        # If trigger type is 'TMS', read trigger from matdatum and return trigger times for trigIndices
        trigger = _read_trigger(matdatum)
        return trigger[trigIndices]
    else:
        # If trigger type is not 'TMS', implement random trigger logic here
        ...  # TODO: random trigger implementation


# Function to read trigger times from matdatum
@lru_cache(maxsize=None)
def _read_trigger(matdatum):
    trigChanIdx = matdatum['TrigChan_ind'][0, 0].astype(int) - 1
    refs = matdatum['rawData/trigger'].flatten()
    return matdatum[refs[trigChanIdx]].flatten()[::2] * 1e3


# Function to check trigger numbers
def _check_trigger_numbers(matdata, blocksinfo) -> None:
    epochIndices = blocksinfo.index.unique().to_numpy()
    for epochIndex in epochIndices:
        trigger = _read_trigger(matdata[epochIndex])
        assert blocksinfo.loc[epochIndex, 'no. of Trigs'].sum() == len(trigger), \
            f'no. of triggers in epoch {epochIndex} does not match with mat-data ({len(trigger)})'
        # matdata[epochIndex][matdata[epochIndex]['CombiMCD_fnames'].flatten()[0]].tobytes().decode('utf-16')


# Function to read MSO data
@lru_cache(maxsize=None)
def _read_MSO(matdatum) -> np.ndarray:
    refs = matdatum['blockInfo/MSO'].flatten()
    mso = [matdatum[i].flatten().tobytes().decode('utf-16') for i in refs]
    return np.array(
        [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else 0 for item in mso])


# Function to expand rows of a blocksinfo record (used by explode method of pandas Dataframe)
def _rowExpander(s):
    ret = list()
    indices = s.index
    for idx in indices:
        match idx:
            case 'MSO ' | 'no. of Trigs' | 'Stimpulses':
                ret.append(s['MSO_order'][idx].to_list())
            case 'Filename':
                ret.append(s[idx].split(','))
            case _:
                ret.append(s[idx])

    return pd.Series(ret, index=s.index)


# Function to check and sort MSO order in blocksinfo to match the actual recording order
def _check_and_sort_mso_order(matdata, blocksinfo) -> None:
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
                df = df.apply(_rowExpander, axis=1).explode(
                    ['MSO ', 'no. of Trigs', 'Stimpulses', 'Filename', 'Queries ', 'Comments', 'Time'])
                df[blocksinfo.index.names] = pd.DataFrame([epochIndex] * df.shape[0])
                argsort = [np.nonzero(df.Filename.str.fullmatch(item))[0][0] for item in fnames]
                df = df.iloc[argsort].set_index(blocksinfo.index.names)
                df.drop(columns=['TrigIndices', 'MSO_order'], inplace=True)
                blocksinfo.loc[epochIndex, :] = _edit_blocksinfo(df.copy(deep=False), 'TrigIndices')


# Function to concatenate a column to blocksinfo with provided value
def _concat_blocksinfo(blocksinfo: pd.DataFrame, colName: str, value: Optional[Any] = None) -> None:
    """
    Adds a new column to the passed DataFrame

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


# Function to combine 'blocksinfo' rows that fall into the same group
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
            case 'Queries ' | 'Comments' | 'Time':
                ret.append(list(subf[col]))

    return pd.Series(ret, index=cols)


def _edit_blocksinfo(blocksinfo: pd.DataFrame, cond: str) -> pd.DataFrame:
    """
    Edits blocksinfo so that each row now pertains to a unique experimental condition. During this process additional
    information needs to be appended in a column, whose name has to be provided.
    Note: As a side effect extra columns will be appended

    Parameters
    ----------
    blocksinfo:     MultiIndex-ed [DataFrame] containing block information
    cond:           name of additional info appended to dataframe

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

    gpExcludeCols = ['MSO ', 'no. of Trigs', 'Stimpulses', 'Filename', 'TrigIndices', 'Depth_int', 'MSO_order',
                     'Queries ', 'Comments', 'Time']
    for col in ['Queries ', 'Comments', 'Time']:
        if col not in blocksinfo.columns:
            blocksinfo[col] = 'none'

    gp = blocksinfo.groupby(blocksinfo.index.names + list(np.setdiff1d(blocksinfo.columns, gpExcludeCols)),
                            group_keys=True,
                            sort=False)
    return (gp.apply(lambda x: _rowCombiner(x, gpExcludeCols)).
            reset_index(level=list(np.setdiff1d(blocksinfo.columns, gpExcludeCols)))
            [blocksinfo.columns])


# Function to merge selection parameters of two instances of analysis parameters so that it adheres to prescribed format
def merge_selectionParams(lSelParams: dict, rSelParams: dict, kind='Outer') -> dict:
    def merge_outer(lvalue, rvalue):
        if lvalue is not None and rvalue is not None:
            # If both values are not None, return the result of set operation 'or'
            if isinstance(lvalue, tuple):
                return tuple(set(lvalue) | set(rvalue)) if isinstance(rvalue, tuple) else tuple(set(lvalue) | {rvalue})
            else:
                return tuple({lvalue} | set(rvalue)) if isinstance(rvalue, tuple) else tuple({lvalue} | {rvalue})
        else:
            # If either value is None, return the non-None value
            return lvalue if rvalue is None else rvalue

    # Generator function to get paths in a nested dictionary
    def paths(tree, cur=()):
        # If tree is not a dictionary, yield the current path
        if not isinstance(tree, dict):
            yield cur
        else:
            # Iterate through key-value pairs in the dictionary
            for k, s in tree.items():
                # Recursively yield paths for nested dictionaries
                for p in paths(s, cur + (k,)):
                    yield p

    # Function to set value in nested dictionary at specified path
    def set_nested_dict(tree, p, value):
        # Traverse nested dictionaries and set value at specified path
        reduce(lambda d, k: d.setdefault(k, {} if k != p[-1] else value), p, tree)
        pass

    # Initialize merged dictionary
    merged = dict()

    # Merge based on kind of merge
    match kind:
        # For 'Outer' merge
        case 'Outer':
            # Iterate through paths in lSelParams
            for path in paths(lSelParams):
                if path[0] in {'Epoch'} | COLS_WITH_STRINGS:
                    # Merge values from lSelParams and rSelParams at current path
                    set_nested_dict(merged, path, merge_outer(reduce(lambda d, k: d.get(k, None), path, lSelParams),
                                                              reduce(lambda d, k: d.get(k, None), path, rSelParams)))
                else:
                    # Set value from lSelParams at current path in merged dictionary
                    set_nested_dict(merged, path, reduce(lambda d, k: d.get(k, None), path, lSelParams))

            # Add remaining keys from rSelParams to merged dictionary
            for key, subtree in rSelParams.items():
                if key not in merged.keys():
                    merged[key] = subtree

        # For other kinds of merge
        case _:
            raise NotImplementedError(f'merging of selectionParams is not implemented for kind=\'{kind}\'')

    # Return merged dictionary
    return merged


class LateComponent(object):
    """LateComponent

    Class dealing with analysis of late activity

    """

    # Available methods for computing delay
    _methodsForComputingDelay = {'threshold_crossing', 'starting_point_derived_from_slope', 'peak_as_delay'}
    # Default method for computing delay
    _defaultMethodForComputingDelay = 'starting_point_derived_from_slope'
    # Time point for separating early and late activity (in msecs)
    earlyLateSeparationTimePoint = 5  # in msecs

    def __init__(self, method=None):
        # Initialize with default method if not specified
        self.delayMethod = LateComponent._defaultMethodForComputingDelay if method is None else method

    @property
    def delayMethod(self):
        return self._delayMethod

    @delayMethod.setter
    def delayMethod(self, method):
        # Set delayMethod property using the value of the 'method' parameter
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

        # Choose the appropriate function based on the delay method
        match self.delayMethod:
            case 'baseline_crossing':
                fcn = self.threshold_crossing
            case 'starting_point_derived_from_slope':
                fcn = self.starting_point_derived_from_slope
            case 'peak_as_delay':
                fcn = self.peak_as_delay

        # Compute the sampling interval
        dt = ps_T[1] - ps_T[0]
        # Convert time (for separating early and late activity) to index
        earlyLateSeparationIdx = (self.earlyLateSeparationTimePoint - ps_T[0]) / dt
        delays = np.empty(shape=0, dtype=float)
        for colIdx in range(meanPSFR.shape[1]):
            # Determine the minimum peak height for peak detection (although a threshold value of 10 may seem arbitrary,
            # it is appropriate for this dataset)
            minPeakHeight = max(1.5 * meanBaselineFR[0, colIdx], 10)
            # Find peaks in peristimulus activity
            peaks, _ = scipy.signal.find_peaks(meanPSFR[:, colIdx], height=minPeakHeight, width=minPeakWidth / dt)
            # Filter late peaks and compute delay for each channel
            latePeaksIdx = np.argwhere(peaks > earlyLateSeparationIdx)

            # Compute delay using the selected function above
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
        # Compute delay as the threshold crossing time point
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
        # Compute delay as the starting point of the trajectory to the peak, derived using slope at inflection point
        df1 = np.diff(waveform, n=1)
        peaks, _ = scipy.signal.find_peaks(df1)
        shadowPeak = find_shadowPeak(peakIdx, dt, waveform, minPeakWidth, scipy.signal.find_peaks(-df1)[0])
        if shadowPeak >= earlyLateSeparationIdx and waveform[shadowPeak] > minPeakHeight:
            peakIdx = shadowPeak
        elif ~np.isnan(shadowPeak) and waveform[shadowPeak] > waveform[peakIdx] and (peakIdx - shadowPeak) * dt < 7.0:
            # if preceding peak height is bigger
            inflectionIdx = peaks[np.argwhere(peaks < shadowPeak)].max()
            return (offset
                    + (peaks[peaks < peakIdx].max() + 0.5 -
                        ((waveform[inflectionIdx:inflectionIdx + 2].mean() - baselineFR) / df1[inflectionIdx])) * dt)
        inflectionIdx = peaks[np.argwhere(peaks < peakIdx)].max()
        return (offset
                + (inflectionIdx + 0.5 -
                   ((waveform[inflectionIdx:inflectionIdx + 2].mean() - baselineFR) / df1[inflectionIdx])) * dt)

    @staticmethod
    @delay_deco
    def peak_as_delay(peakIdx, dt, offset):
        # Compute delay as the peak offset
        return (peakIdx * dt) + offset


# Function to find shadow peak (the peak that is smaller and appears earlier but closer than separation distance
# to the larger peak)
def find_shadowPeak(peakIdx, dt, waveform, minPeakWidth, troughs):
    troughs = troughs[troughs < peakIdx]
    if troughs.size > 0:
        return troughs[-1] + 1 - waveform[troughs[-1] - range(-1, np.rint(minPeakWidth / dt).astype(int))].argmax()
    return np.nan


# Function that compares pd.Series to number using a string containing that numeral and also the comparison symbol
def comparator(ser, string):
    # Check if the string pattern matches one of the given symbols
    if re.match('<=', string):
        # Return True for elements in 'ser' that are less than or equal to the specified numeral
        return ser <= np.float_(re.sub('<=', '', string))
    elif re.match('<', string):
        # Return True for elements in 'ser' that are less than specified numeral
        return ser < np.float_(re.sub('<', '', string))
    elif re.match('>=', string):
        # Return True for elements in 'ser' that are greater than or equal to the specified numeral
        return ser >= np.float_(re.sub('>=', '', string))
    elif re.match('>', string):
        # Return True for elements in 'ser' that are greater than the specified numeral
        return ser > np.float_(re.sub('>', '', string))
    elif re.match('==', string):
        # Return True for elements in 'ser' that are equal to the specified numeral
        return ser == np.float_(re.sub('==', '', string))
    elif re.match('!=', string):
        # Return True for elements in 'ser' that are equal to the specified numeral
        return ser != np.float_(re.sub('!=', '', string))
    else:
        raise ValueError(f'String \'{string}\' in tms.analysis_params has invalid comparator for floats. \n'
                         f'Select one from this list [< : <= : >= : > : == : !=] and without leading/trailing spaces')

