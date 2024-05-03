import numpy as np
import numba as nb
import pandas as pd
from statistics import mean, mode
import os
from typing import Optional, Any
from itertools import zip_longest
from functools import lru_cache, cached_property
import scipy

from lib.matpy import MATfile
import lib.helper_tms_tg as th
from lib.constants import LAYERS, REGIONS, EPOCHISOLATORS, COLS_WITH_STRINGS, COLS_WITH_NUMS
from lib.dataanalysis import peristim_firingrate, peristim_timestamp


def _filter_blocks_helper(
        blocksinfo, analysis_params, fIdx=None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filters MultiIndex-ed blocksinfo <DataFrame> using 'selectionParams' in analysis_params

    Parameters
    ----------
    blocksinfo : pd.DataFrame
        MultiIndex-ed DataFrame containing block information.
    analysis_params : dict
        Dictionary with parameters for filtering blocksinfo.
    fIdx : pd.Series, optional
        MultiIndex-ed boolean array, represents filter indices.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Filtered MultiIndex-ed blocksinfo, MultiIndex-ed boolean array (filter indices).

    """
    # Initialize Index for selecting (booleanIndexing) blocks using criterion in ['selectionParams']
    if fIdx is None:
        fIdx = blocksinfo['MSO '] == blocksinfo['MSO ']

    # Change the truth values of Index by doing string comparison on dataframe.Index
    epochIndices = blocksinfo.index.to_frame()
    for item in EPOCHISOLATORS:
        if (strings := analysis_params['selectionParams']['Epoch'][item]) is not None:
            if isinstance(strings, tuple):
                fIdx &= epochIndices[item].str.contains('|'.join(strings))
            else:
                fIdx &= epochIndices[item].str.contains(strings)

    # Change the truth values of Index by doing floating point comparison on dataframe columns
    selectCols = analysis_params['selectionParams'].keys() & COLS_WITH_NUMS
    for col in selectCols:
        fIdx &= th.comparator(blocksinfo[col], analysis_params['selectionParams'][col])

    # Change the truth values of Index by doing string comparison on dataframe columns
    selectCols = analysis_params['selectionParams'].keys() & COLS_WITH_STRINGS
    for col in selectCols:
        strings = analysis_params['selectionParams'][col]
        if isinstance(strings, tuple):
            fIdx &= blocksinfo[col].str.contains('|'.join(strings), case=False)
        else:
            fIdx &= blocksinfo[col].str.contains(strings, case=False)

    # Convert nan values to False
    fIdx = fIdx == 1

    return blocksinfo.loc[fIdx, :], fIdx


class FilterBlocks(object):
    """
    Descriptor class for filtering blocksinfo <DataFrame> using analysis parameters.

    Attributes:
    - cache: Cached result of filtering operation for optimization.

    Methods:
    - __init__: Initializes the descriptor with a cache attribute set to None.
    - __get__: Retrieves the filtered DataFrame and filter indices. If cache is None, performs filtering and caches
               the result before returning it.
    - __set__: Sets the cache attribute to the given value.
    - __call__: Callable method for performing filtering operation directly.

    """

    def __init__(self):
        """
        Initializes the FilterBlocks descriptor with a cache attribute set to None.
        """
        self.cache = None

    def __get__(self, obj, objType):
        """
        Retrieves the filtered DataFrame and filter indices.

        If cache is None, performs filtering and caches the result.

        Parameters:
        - obj: Instance of the class where the descriptor is defined.
        - objType: Type of the object where the descriptor is defined.

        Returns:
        - tuple: Filtered DataFrame and filter indices.
        """
        if self.cache is None:
            self.cache = _filter_blocks_helper(obj.blocksinfo, obj.analysis_params, fIdx=None)
        return self.cache[0], self.cache[1]

    def __set__(self, obj, value):
        """
        Sets the cache attribute to the given value.

        Parameters:
        - obj: Instance of the class where the descriptor is defined.
        - value: Value to set the cache attribute.
        """
        self.cache = value

    def __call__(self, blocksinfo, analysis_params, filterIndices=None):
        """
        Performs filtering of block information.

        Parameters:
        - blocksinfo: DataFrame containing block information.
        - analysis_params: Parameters for filtering blocksinfo.
        - filterIndices: Optional filter indices used as starting value.

        Returns:
        - tuple: Filtered DataFrame and filter indices.
        """
        return _filter_blocks_helper(blocksinfo, analysis_params, filterIndices)


def hem_CoilDir_selector(blocksinfo, index, hem, cDir):
    """
    Selects rows from blocksinfo DataFrame based on specified hemisphere (hem) and coil direction (cDir).

    Parameters:
    - blocksinfo: DataFrame containing block information.
    - index: Index or indices of rows to consider.
    - hem: Hemisphere to filter for ('RH', 'LH', 'same', or 'opposite').
    - cDir: Coil direction to filter for ('ML' or 'LM').

    Returns:
    - np.ndarray: Boolean array indicating selected rows.
    """

    def coilDir_converter(ser):
        """
        Returns standardized coil direction based on stimulation hemisphere and coil direction values.

        Parameters:
        - ser: Series containing 'StimHem' and 'CoilDir' values.

        Returns:
        - str: Converted current direction.
        """
        if ser['StimHem'] == 'RH' and any(ser['CoilDir'] == np.array(['L-R', 'LR'])):
            return 'ML'
        elif ser['StimHem'] == 'RH' and any(ser['CoilDir'] == np.array(['R-L', 'RL'])):
            return 'LM'
        elif ser['StimHem'] == 'LH' and any(ser['CoilDir'] == np.array(['L-R', 'LR'])):
            return 'LM'
        elif ser['StimHem'] == 'LH' and any(ser['CoilDir'] == np.array(['R-L', 'RL'])):
            return 'ML'
        else:
            return ser['CoilDir']

    # Get indices of columns 'StimHem' and 'RecHem'
    stimHemIdx = np.nonzero(np.array(blocksinfo.columns) == 'StimHem')[0]
    recHemIdx = np.nonzero(np.array(blocksinfo.columns) == 'RecHem')[0]

    # Apply coilDir_converter to create a standardized coil direction column
    coilDir = blocksinfo.loc[:, ['StimHem', 'CoilDir']].apply(coilDir_converter, axis=1)

    # Select rows based on hemisphere and coil direction
    match hem:
        case 'RH':
            return np.logical_and(blocksinfo.iloc[index, stimHemIdx].to_numpy().ravel() == 'RH',
                                  coilDir.iloc[index].to_numpy() == cDir)
        case 'LH':
            return np.logical_and(blocksinfo.iloc[index, stimHemIdx].to_numpy().ravel() == 'LH',
                                  coilDir.iloc[index].to_numpy() == cDir)
        case 'opposite':
            return np.logical_and(blocksinfo.iloc[index, stimHemIdx].to_numpy().ravel()
                                  != blocksinfo.iloc[index, recHemIdx].to_numpy().ravel(),
                                  coilDir.iloc[index].to_numpy() == cDir)
        case _:
            return np.logical_and(blocksinfo.iloc[index, stimHemIdx].to_numpy().ravel()
                                  == blocksinfo.iloc[index, recHemIdx].to_numpy().ravel(),
                                  coilDir.iloc[index].to_numpy() == cDir)


# Select a block from blocksinfo DataFrame based on the specified epoch and amplitude criteria.
def block_selector(blocksinfo: pd.DataFrame, epoch: tuple, amplitude: Optional[str] = 'MT') \
        -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Find out the index of the block that matches the given amplitude for a particular epoch. In case the epoch
    contains multiple blocks with same amplitude as it happens when injections are done, then select the one prior
    to injection, if it exists, otherwise select the last 'Post' injection block. In the additional case, where the
    epoch contains multiple blocks with same amplitude due to different combinations of StimHem and CoilDir, select the
    one with CoilDir == 'ML' and StimHem == RecHem.

    Parameters
    ----------
    - blocksinfo:     MultiIndex-ed DataFrame containing block information.
    - epoch:          Tuple specifying the epoch criteria
    - amplitude:      Optional string specifying the amplitude criteria ('MT', 'minimum', 'maximum'). Default is 'MT'.

    Returns
    -------
    - np.ndarray:   index [0-Dim np.ndarray] of the selected epoch
    """

    # Initialize FilterBlocks instance
    fb = FilterBlocks()

    # Determine the selection criteria based on amplitude
    match amplitude:
        case 'MT':
            mtCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])),
                                          'MT': '==1'}}
        case 'minimum':
            mtCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])),
                                          'MT': f'=={blocksinfo.loc[epoch, "MT"].min()}'}}
        case 'maximum':
            mtCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])),
                                          'MT': f'=={blocksinfo.loc[epoch, "MT"].max()}'}}
        case _:
            raise ValueError(f'epoch_selector() does not except argument \'{amplitude}\' as a value '
                             f'for parameter "amplitude"')

    # Get boolean indices of the blocks meeting the MT criteria
    _, mtIdx = fb(blocksinfo, mtCond)

    # Identify injection types present in the dataset
    injTypesPresent = list({'Skin-Injection', 'TG-Injection ', 'TGOrifice'} & set(blocksinfo.columns))

    # Initialize boolean index for 'Post' conditions
    postIdx = mtIdx.apply(lambda x: False)

    # Iterate over injection types present and update 'Post' index
    for injType in injTypesPresent:
        postIdx |= fb(blocksinfo, {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])),
                                                       injType: 'Post'}})[1]

    # Get index of blocks meeting all criteria
    index = np.nonzero((blocksinfo.index == epoch) & mtIdx.to_numpy() & ~postIdx.to_numpy())[0]

    # Handle cases where no blocks meet the criteria
    if not any(index):
        # Here its considered that only one injection condition was experimentally tested at a time, i.e., no overlaps,
        # so first extract it. Then either the last 'Post' condition pertaining to stimulus 'amplitude' is used as index
        # or the last 'Post' condition under the criteria CoilDir == 'ML' and StimHem == RecHem is used as the index
        injTypesPostRelation = blocksinfo.loc[epoch, injTypesPresent].apply(
            lambda x: x.str.contains('post', case=False)).any()
        injWithPost = injTypesPostRelation.index[injTypesPostRelation.values]
        if len(injWithPost) > 0:
            postT = blocksinfo[injWithPost.item()].str.extract('(\d+)').astype('float64')
            postT.loc[list(set(blocksinfo.index.unique()) - {epoch})] = 0.0
            postT[~mtIdx.to_numpy()] = 0.0
            bIdx = hem_CoilDir_selector(blocksinfo, np.nonzero(postT)[0], 'same', 'ML')
            if any(bIdx):
                postT.iloc[np.nonzero(postT)[0][~bIdx]] = 0.0
            index = np.nonzero(postT == postT.max())[0]

    # Handle cases where multiple blocks meet the criteria
    elif len(index) > 1:
        # Find the index that corresponds to criteria CoilDir == 'ML' and StimHem == RecHem
        bIdx = hem_CoilDir_selector(blocksinfo, index, 'same', 'ML')
        if any(bIdx):
            index = index[bIdx]
        else:
            # if criteria CoilDir == 'ML' and StimHem == RecHem is non-existent, pick the first block.
            index = index[0]

    return index


class TMSTG(object):
    """TMSTG

    Enables easy access to data using their locations and importantly eases running analysis on that data

    Args
    ----------
    matdata: pd.Series
        MultiIndex-ed collection of MATdata objects that enable easy access to data on disk
    blocksinfo: pd.DataFrame
        MultiIndex-ed DataFrame containing block information.

    Note:   Do not instantiate this class directly. Instead, use...
            TMSTG.load(groupinfofilePath, ).
            @params: groupinfofilePath:  file containing paths to all locations having data of interest


    Class attributes
    ----------------
    analysis_params:        parameters for data selection and data analysis
    filter_blocks:          filtered blocks pertaining to selection done using analysis parameters
    late_comp:              derived latency of late activity component


    Attributes
    ----------
    psts:                   {cached property}    trial-wise organized spike time-stamps


    Methods
    -------
    load:                   Initializes a TMSTG object using groupinfofilePath.
    compute_firingrate:     computes trialwise (peristimulus) firing rate activity.
    avg_FR_per_neuron:      computes the average of trialwise firing rate activity.
    do_multi_indexing:      Multi-indexes dataframe having block infos (that was originally retrieved from infofile).
    singleUnitsSpikeTimes:  Callable method for performing filtering operation directly.

    """

    # Default analysis parameters for the TMSTG class
    _default_analysis_params = {
        'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ]))},
        'TMSArtifactParams': {'timeWin': (-0.3, 0.5)},
        'peristimParams': {'smoothingParams': {'win': 'gaussian', 'width': 1.5, 'overlap': 1 / 3},
                           'timeWin': (-50.0, 100.0),
                           'trigger': 'TMS',
                           'baselinetimeWin': (-50.0, -1.0)}}

    # Initialize descriptors: analysis parameters, filter blocks, and late component
    analysis_params = th.AnalysisParams(_default_analysis_params)
    filter_blocks = FilterBlocks()
    late_comp = th.LateComponent()

    def __init__(self, matdata=None, blocksinfo=None) -> None:
        # Initialize TMSTG object with MATdata objects and blocks information
        self.matdata: Optional['pd.Series[mp.MATdata]'] = matdata.sort_index()
        self.blocksinfo: Optional[pd.DataFrame] = blocksinfo.sort_index()

        # Check that trigger numbers listed in blocksinfo matches with actual data read from MATdata objects
        # and sort MSO order of blocks in blocksinfo to match the recording order
        th._check_trigger_numbers(self.matdata, self.blocksinfo)
        th._check_and_sort_mso_order(self.matdata, self.blocksinfo)

    @classmethod
    def load(cls, groupinfofilePath: str) -> 'TMSTG':
        """
        create a TMSTG object using paths to all locations containing data of interest

        Parameters
        ----------
        groupinfofilePath:  path to a group infofile (this file contains paths to all locations having data of interest)

        Returns
        -------
        an instance of TMSTG class
        """

        # Load group information from an Excel file and create TMSTG object
        groupinfo = pd.read_excel(groupinfofilePath).dropna()
        groupMatlabfnames = pd.Series()
        groupBlocksinfo = pd.DataFrame()

        # Iterate over each group in the groupinfo Excel file
        for i in groupinfo.index:
            # Get file paths for animal infofile and MATLAB files
            animalInfofilePath = [groupinfo.loc[i, 'Folder'] + '\\' + f
                                  for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.xlsx')]
            animalMatlabfnames = pd.Series(groupinfo.loc[i, 'Folder'] + '\\' + f
                                           for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.mat'))

            # Read blocks information from animal infofile
            if len(animalInfofilePath) > 0:
                blocksinfo = pd.read_excel(animalInfofilePath[0]).drop(
                    columns=['Unnamed: 11', 'Queries'], errors='ignore').dropna(axis=0, how='all')
                th._concat_blocksinfo(blocksinfo, 'Animal', str(groupinfo.loc[i, 'Animal']))
                blocksinfo = cls.do_multi_indexing(blocksinfo, animalMatlabfnames)
                blocksinfo = th._edit_blocksinfo(blocksinfo.copy(deep=False), 'TrigIndices')
                cls._sort_filelist(animalMatlabfnames, blocksinfo)
                groupMatlabfnames = pd.concat([groupMatlabfnames, animalMatlabfnames])
                groupBlocksinfo = pd.concat([groupBlocksinfo, blocksinfo])

        # Create TMSTG object using MATdata objects and blocks information
        return cls(
            pd.Series([MATfile(fname).read() for fname in groupMatlabfnames], index=groupBlocksinfo.index.unique()),
            groupBlocksinfo)

    @cached_property
    def psts(self) -> list:
        """
        Organizes the time-stamps of spikes around (peri) the stimulus in a trialwise fashion

        Returns
        -------
        List: containing spike time-stamps from each block

        """
        print('psts runs...........')
        selectBlocksinfo, selectBlocksinfoIdx = self.filter_blocks

        psTS = list()
        for epochIndex, blockinfo in selectBlocksinfo.iterrows():
            selectTrigger = th._get_trigger_times_for_current_block(
                self.analysis_params['peristimParams']['trigger'], self.matdata[epochIndex], blockinfo['TrigIndices'])
            timeWindows = selectTrigger[:, np.newaxis] + np.array(self.analysis_params['peristimParams']['timeWin'])
            psTS.append(peristim_timestamp(self.singleUnitsSpikeTimes(epochIndex), timeWindows))

        return psTS

    @lru_cache(maxsize=None)
    def compute_firingrate(self,
                           smoothingWinType,
                           smoothingWinWidth,
                           smoothingWinOverlap,
                           timeWinLeftEndpoint,
                           timeWinRightEndpoint,
                           triggerType) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Compute (trialwise) peri-stimulus firing rate activity

        Parameters
        ----------
        smoothingWinType: Type of smoothing window.
        smoothingWinWidth: Width of the smoothing window.
        smoothingWinOverlap: Overlap of the smoothing window.
        timeWinLeftEndpoint: Left endpoint of the time window.
        timeWinRightEndpoint: Right endpoint of the time window.
        triggerType: Type of trigger.

        Returns
        -------
        Tuple[list, np.ndarray]:
        the first item is a list (length equaling the number of blocks after filtering)
        of computed peri-stimulus firing rate activity
        (each item in the list has a size of R X T X N; R equals no. of trials (repetitions) in a block,
        T equals no. of time points for computing psfr, and N equals no. of neurons present),
        and the second item is an 1-D array of time index
        """

        # Create a tuple using input parameters
        params = (smoothingWinType, smoothingWinWidth, smoothingWinOverlap,
                  timeWinLeftEndpoint, timeWinRightEndpoint, triggerType)

        print('compute_firingrate runs...........')

        # If compute_firingrate in running for first time create a new column 'psActivity' in blocksinfo
        if 'psActivity' not in self.blocksinfo.columns:
            self.blocksinfo['psActivity'] = None
            self.filter_blocks = None

        # Get the index of 'psActivity' column
        psActivityIdx = np.nonzero(self.blocksinfo.columns == 'psActivity')[0][0]

        # Get the filter indices for current analysis parameters
        selectBlocksinfo, selectBlocksinfoIdx = self.filter_blocks

        # Initialize lists to store firing rates and time points
        ps_FR, ps_T = list(), np.array([])

        # Iterate over selected (filtered) blocks and corresponding indices
        for (epochIndex, blockinfo), idx in zip(selectBlocksinfo.iterrows(), np.nonzero(selectBlocksinfoIdx)[0]):
            # If firing rate for given parameters is already computed, append it to the returned list
            if (self.blocksinfo.iat[idx, psActivityIdx] is not None
                    and params in (cell := self.blocksinfo.iat[idx, psActivityIdx]).keys()):
                tmp_ps_FR, ps_T = cell[params]
                ps_FR.append(tmp_ps_FR)
                continue

            # Get trigger times for the current block
            selectTrigger = th._get_trigger_times_for_current_block(triggerType,
                                                                    self.matdata[epochIndex], blockinfo['TrigIndices'])
            # Define time windows based on the trigger times and specified time window
            timeWindows = selectTrigger[:, np.newaxis] + np.array([timeWinLeftEndpoint, timeWinRightEndpoint])

            # Compute firing rate at defined time windows using given smoothing parameters
            tmp_ps_FR, ps_T = peristim_firingrate(self.singleUnitsSpikeTimes(epochIndex),
                                                  timeWindows,
                                                  {'win': smoothingWinType,
                                                   'width': smoothingWinWidth,
                                                   'overlap': smoothingWinOverlap})
            # Update time points
            ps_T += timeWinLeftEndpoint

            # Cache the computed firing rate for future use
            if (cell := self.blocksinfo.iat[idx, psActivityIdx]) is not None:
                cell[params] = [tmp_ps_FR, ps_T]
            else:
                self.blocksinfo.iat[idx, psActivityIdx] = {params: [tmp_ps_FR, ps_T]}

            # Reset filter_blocks
            self.filter_blocks = None

            # Append computed firing rate to the list
            ps_FR.append(tmp_ps_FR)

        return ps_FR, ps_T

    @staticmethod
    def do_multi_indexing(blocksinfo: pd.DataFrame, matlabfnames: pd.Series) -> pd.DataFrame:
        """
        Multi-indexes the passed in DataFrame

        Parameters
        ----------
        blocksinfo:     pd.DataFrame
            Experimental block information.
        matlabfnames:   pd.Series
            list of full-file paths to MATLAB files.

        Returns
        -------
        DataFrame with multi-indexing applied.
        """

        # Create a copy of the blocksinfo DataFrame
        df = blocksinfo.copy()

        # Remove suffix 'µm' or 'μm' from 'Depth' column and convert to integer
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('µm')
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('μm')
        df['Depth_int'] = np.int_(df.loc[:, 'Depth'].to_list())

        # Convert 'no. of Trigs' column to integer
        df['no. of Trigs'] = np.int_(df.loc[:, 'no. of Trigs'].to_list())

        # Assign values to the first epoch isolator based on 'RecArea' and 'REGIONS' mapping
        for key in REGIONS:
            df.loc[df['RecArea '].str.contains('|'.join(REGIONS[key])), EPOCHISOLATORS[1]] = key
        recRegions = df.loc[:, EPOCHISOLATORS[1]].unique()

        # Assign values to the second epoch isolator based on 'LAYERS' mapping
        df.loc[:, EPOCHISOLATORS[2]] = np.nan
        corticalRegions = set(recRegions) & set(LAYERS.keys())
        for cRegion in corticalRegions:
            for key in LAYERS[cRegion]:
                df.loc[(LAYERS[cRegion][key][0] <= df['Depth_int'])
                       & (df['Depth_int'] < LAYERS[cRegion][key][1])
                       & df['Region'].str.contains(cRegion), EPOCHISOLATORS[2]] \
                    = key

        # Assign values to the third epoch isolator based on 'StimHem' and 'RecHem'
        if not df['StimHem'].isna().all():
            df.loc[df['StimHem'] == df['RecHem'], EPOCHISOLATORS[3]] = 'same'
            df.loc[df['StimHem'] != df['RecHem'], EPOCHISOLATORS[3]] = 'opposite'
        else:
            df.loc[:, EPOCHISOLATORS[3]] = np.nan

        # Assign values to the fourth epoch isolator based on 'Filename'
        if all(map(lambda x: isinstance(x, str), df['Filename'])):
            df.loc[df['Filename'].str.contains('con'), EPOCHISOLATORS[4]] = 'contra'
            df.loc[df['Filename'].str.contains('ips'), EPOCHISOLATORS[4]] = 'ipsi'
        elif df['Filename'].isna().all():
            df.loc[:, EPOCHISOLATORS[4]] = np.nan
        else:
            raise ValueError(f'The infofile column \'Filename\' cannot be half empty '
                             f'which is the case for animal {df["Animal"][0]}')

        # Assign values to 'Movement' column based on 'Mov' column if necessary
        if not (matlabfnames.str.contains('|'.join(('same', 'opposite')))
                | matlabfnames.str.contains('|'.join(('con', 'ips')))).any():
            df.loc[:, 'Movement'] = df.loc[:, 'Mov'].copy()
            df.loc[:, EPOCHISOLATORS[3:5]] = np.nan

        # Fill NaN values with 'none' and set multi-index
        df.fillna('none', inplace=True)
        df.set_index(EPOCHISOLATORS, inplace=True)

        return df

    def avg_FR_per_neuron(self, squeezeDim=True) -> tuple[np.ndarray | list, np.ndarray, np.ndarray | list, np.ndarray]:
        """
        Compute the average of peristimulus firing rate activity for each neuron

        Parameters
        ----------
        squeezeDim: [Flag]  True:   returns 2-D array
                            False:  returns 3-D array (mean computed over axis=0)

        Returns
        -------
        Tuple[., ., ., .] -->

        array[Time X N] - average peristimulus activity per neuron,
        array[1D]       - Time points of peristimulus activity,
        array[1 X N]    - average baseline firing rate per neuron,
        array[1D]       - mid-time point of baseline firing rate

        or

        list of arrays[[1 X Time X N], ]    - average peristimulus activity per neuron,
        array[1D]                           - Time points of peristimulus activity,
        list of arrays[[1 X N], ]           - average baseline firing rate per neuron,
        array[1D]                           - mid-time point of baseline firing rate
        """

        # noinspection DuplicatedCode
        # Compute peristimulus firing rate and baseline firing rate activity
        ps_FR, ps_T = self.compute_firingrate(*self.analysis_params['peristimParams']['smoothingParams'].values(),
                                              *self.analysis_params['peristimParams']['timeWin'],
                                              self.analysis_params['peristimParams']['trigger'])
        ps_baseline_FR, ps_baseline_T = (
            self.compute_firingrate('rectangular',
                                    np.diff(self.analysis_params['peristimParams']['baselinetimeWin']).item(0),
                                    0.0,
                                    mean(self.analysis_params['peristimParams']['baselinetimeWin']),
                                    self.analysis_params['peristimParams']['baselinetimeWin'][1],
                                    self.analysis_params['peristimParams']['trigger']))

        if squeezeDim:
            return np.asfortranarray(np.concatenate([block_psfr.mean(axis=0) for block_psfr in ps_FR], axis=1)), \
                ps_T, \
                np.concatenate([block_bsfr.mean(axis=0) for block_bsfr in ps_baseline_FR], axis=1), \
                ps_baseline_T
        else:
            return [block_psfr.mean(axis=0, keepdims=True) for block_psfr in ps_FR], \
                ps_T, \
                [block_bsfr.mean(axis=0, keepdims=True) for block_bsfr in ps_baseline_FR], \
                ps_baseline_T

    def stats_is_signf_active(self,
                              trialwise_PS_FR: Optional[list[np.ndarray]] = None,
                              trialwise_BL_FR: Optional[list[np.ndarray]] = None) -> 'pd.Series[list[bool]]':

        """
        Compute whether neurons are significantly active.

        Parameters:
            trialwise_PS_FR: Optional list of trial-wise peri-stimulus firing rates.
            trialwise_BL_FR: Optional list of trial-wise baseline firing rates.

        Returns:
            pd.Series containing boolean values indicating whether neurons are significantly active.
        """
        # Compute trial-wise post-stimulus and baseline firing rates if not provided
        if trialwise_PS_FR is None:
            trialwise_PS_FR, _ = self.compute_firingrate('rectangular', 49.0, 0.0, 25.5, 50.0,
                                                         self.analysis_params['peristimParams']['trigger'])
        if trialwise_BL_FR is None:
            trialwise_BL_FR, _ = self.compute_firingrate('rectangular', 49.0, 0.0, -25.5, -1,
                                                         self.analysis_params['peristimParams']['trigger'])

        # Get unique epoch names of selected blocks
        selectBlocksinfo, selectBlocksinfoIdx = self.filter_blocks
        epochs = selectBlocksinfo.index.unique()

        # Compute whether neurons are significantly active at MT stimulation (if non-existent than Max stimulation,
        # for silencing conditions the last MT stimulation)
        activeNeus = pd.Series(np.empty(shape=epochs.shape), index=epochs)
        for epoch in epochs:
            index = block_selector(selectBlocksinfo, epoch, amplitude='MT')
            assert len(index) < 2, f'epoch {epoch} has more than one MT cond, not correct'
            if len(index) == 0:
                index = block_selector(selectBlocksinfo, epoch, amplitude='maximum')
                assert len(index) < 2, f'epoch {epoch} has more than one MT cond, not correct'
            postStimFR = trialwise_PS_FR[index.item()]
            baselineFR = trialwise_BL_FR[index.item()]
            activeNeus[epoch] = scipy.stats.ttest_ind(postStimFR[:, 0, :],
                                                      baselineFR[:, 0, :],
                                                      axis=0,
                                                      alternative='greater').pvalue < 0.05

        return activeNeus

    @lru_cache(maxsize=None)
    def singleUnitsSpikeTimes(self, epochIndex) -> 'nb.typed.List':
        """
        Retrieve spike times for single units from specific epochs.

        Args:
            epochIndex: Index of the epoch.

        Returns:
            List of spike times of single units.
        """
        # Retrieve spike times
        multiUnitSpikeTimes: np.ndarray = self.matdata.loc[epochIndex]['SpikeModel/SpikeTimes/data'].flatten()
        refs = self.matdata.loc[epochIndex]['SpikeModel/ClusterAssignment/data'].flatten()
        singleUnitsIndices = [self.matdata.loc[epochIndex][i].flatten().astype(int) - 1 for i in refs]

        # Remove spikes within TMS artifact time window if specified
        if self.analysis_params['TMSArtifactParams'] is not None:
            singleUnitsIndices = self._remove_spikes_within_TMSArtifact_timeWin(multiUnitSpikeTimes, singleUnitsIndices,
                                                                                epochIndex)

        # Convert to typed List
        sUSpikeTimes = nb.typed.List()
        [sUSpikeTimes.append(multiUnitSpikeTimes[sU_indices]) for sU_indices in singleUnitsIndices]
        return sUSpikeTimes

    @staticmethod
    def _sort_filelist(matlabfnames, blocksinfo) -> None:
        """
        Sorts the order of matlabf<ile>names in the Series to be consistent with epoch order in blocksinfo

        Parameters
        ----------
        matlabfnames:   Series of matlabf<ile>names
        blocksinfo:      MultiIndex-ed DataFrame containing block information

        """

        # Function to look for matching file names
        def lookfor_matching_fname(boolArray, *string) -> pd.Series:
            """
            Recursive function to look for matching file names.

            Args:
                boolArray: Boolean array indicating matching conditions.
                *string: Variable number of strings to match.

            Returns:
                Boolean Series indicating matching file names.
            """
            # Look for matching file names recursively
            if (boolArray & matlabfnames.str.contains(string[0], case=False)).any() & (len(string) > 1):
                boolArray &= matlabfnames.str.contains(string[0], case=False)
                return lookfor_matching_fname(boolArray, *string[1:])
            elif (not (boolArray & matlabfnames.str.contains(string[0], case=False)).any()) & (len(string) > 1):
                return lookfor_matching_fname(boolArray, *string[1:])
            elif (boolArray & matlabfnames.str.contains('_' + string[0])).any() & (len(string) == 1):
                return boolArray & matlabfnames.str.contains('_' + string[0])
            else:
                return boolArray

        # Use MultiIndexes of blocksinfo to set the file order
        epochIndices = blocksinfo.index.unique()
        for i, epochIndex in enumerate(epochIndices):
            matchBoolArray = lookfor_matching_fname(np.ones(len(matlabfnames), dtype=bool), *epochIndex)
            j = matchBoolArray.array.argmax()
            matlabfnames.iloc[[i, j]] = matlabfnames.iloc[[j, i]]

        matlabfnames.index = epochIndices
        pass

    def _remove_spikes_within_TMSArtifact_timeWin(self,
                                                  spikeTimes: np.ndarray,
                                                  singleUnitsIndices: list,
                                                  epochIndex: tuple) -> list[np.ndarray]:
        """
        Removes spikes within the closed time-interval specified in 'TMSArtifactParams'

        Parameters
        ----------
        spikeTimes:             Array of multiUnitSpikeTimes
        singleUnitsIndices:     List of single unit indices
        epochIndex:             index of the epoch from which multiUnitSpikeTimes were extracted

        Returns
        -------
        list of single unit indices that fall outside the closed interval specified in 'TMSArtifactParams'
        """

        # Retrieve trigger references and indices
        refs = self.matdata[epochIndex]['rawData/trigger'].flatten()
        trigChanIdx = self.matdata[epochIndex]['TrigChan_ind'][0, 0].astype(int) - 1
        ampTrigIdx = 2

        # Compute time intervals based on amplifier trigger or trigger channel
        if len(ampTrigger := self.matdata[epochIndex][refs[ampTrigIdx]].flatten() * 1e3) \
                >= len(trigger := self.matdata[epochIndex][refs[trigChanIdx]].flatten() * 1e3):
            if len(ampTrigger) > len(trigger):
                # take care of wrong ampTriggers
                ampTrigger = ampTrigger.reshape((ampTrigger.size // 2, 2))
                cond = mode(np.diff(ampTrigger, axis=1)[0])
                i = 0
                while i < ampTrigger.shape[0]:
                    if (ampTrigger[i, 1] - ampTrigger[i, 0]) < (cond - 0.1):
                        ampTrigger = np.delete(ampTrigger, i * 2 + np.array([1, 2]))
                        ampTrigger = ampTrigger.reshape(ampTrigger.size // 2, 2)
                    i += 1
                assert ampTrigger.size == len(trigger), \
                    f'for epoch {epochIndex} length of ampTrigger and tmsTrigger are not the same'

            timeIntervals = (ampTrigger.reshape((ampTrigger.size // 2, 2))
                             + np.array(self.analysis_params['TMSArtifactParams']['timeWin']))

        else:
            # Handle cases where amplifier trigger channel is empty
            print(f'Amplifier trigger channel no. {ampTrigIdx} is empty for epoch {epochIndex},... ')
            print(f'therefore using trigger channel no. {trigChanIdx} to remove spikes around TMSArtifact....')
            print(f'As the width of the trigger for this channel is {np.mean(trigger[1::2] - trigger[::2])} secs, ')
            print(f'it is expanded to match amplifier trig width of {1 + np.mean(trigger[1::2] - trigger[::2])} secs.')
            print('')
            timeIntervals = (trigger.reshape((trigger.size // 2, 2))
                             + (np.array([-0.2, 0.8], dtype=trigger.dtype)
                                + np.array(self.analysis_params['TMSArtifactParams']['timeWin'])))

        # Create mask to remove spikes within time intervals
        mask = np.zeros_like(spikeTimes, dtype=bool)
        for timeInterval in timeIntervals:
            mask |= (timeInterval[0] <= spikeTimes) & (spikeTimes <= timeInterval[1])

        # Get the indices of the mask which represents indices of spikes within time intervals
        mask = mask.nonzero()

        # Remove spikes within time intervals and return it
        return [np.setdiff1d(sU_indices, mask) for sU_indices in singleUnitsIndices]


if __name__ == '__main__':

    # Load TMSTG object using a list containing the locations of the data of individual animals
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)

    # Set analysis parameters for selecting epochs and mean threshold
    tms.analysis_params = {'selectionParams': {'Epoch': {'Region': 'MC', 'Layer': 'L5'}, 'MT': '>=0.9'}}

    # Uncomment the following block to use alternative analysis parameters
    # tms.analysis_params = {'selectionParams': {'Epoch': {'Region': 'thal'}, 'MT': '>=1.2',
    #                                            'RecArea ': ('VPM', 'PO', 'VM', 'VPL')}}

    # Get the index of the active (statistically significant) neurons
    activeNeu = tms.stats_is_signf_active()

    # Filter blocks based on analysis parameters
    selectBlocks, selectBlocksIdx = tms.filter_blocks

    # Flatten activeNeu into a 1D array
    flat_activeNeu = np.empty(shape=0, dtype=bool)
    [flat_activeNeu := np.append(flat_activeNeu, activeNeu[item]) for item in selectBlocks.index]

    # Calculate mean peristimulus firing rate, and baseline firing rate
    meanPSFR, t, meanBaselineFR, _ = tms.avg_FR_per_neuron()

    # Compute the latency of the late activity component
    delays = tms.late_comp.compute_delay(meanPSFR[:, flat_activeNeu],
                                         t,
                                         meanPSFR[t < 0, :].max(axis=0, keepdims=True)[:, flat_activeNeu],
                                         tms.analysis_params['peristimParams']['smoothingParams']['width'] + 0.25)

    # Display delays
    delays

