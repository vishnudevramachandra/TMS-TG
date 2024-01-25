import numpy as np
import numba as nb
import pandas as pd
from statistics import mean, mode
import os
import re
from typing import Optional
from itertools import zip_longest
from functools import lru_cache, cached_property
import scipy

import lib.matpy as mp
import lib.helper_tms_tg as th
from lib.constants import LAYERS, REGIONS, EPOCHISOLATORS, COLS_WITH_STRINGS, COLS_WITH_FLOATS
from lib.dataanalysis import peristim_firingrate, peristim_timestamp


def _filter_blocks_helper(
        blocksinfo, analysis_params, fIdx=None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filters MultiIndex-ed blocksinfo<DataFrame> using 'selectionParams' in analysis_params

    Parameters
    ----------
    blocksinfo:         MultiIndex-ed DataFrame containing block information.
    analysis_params:    Dict with parameters for filtering blocksinfo
    fIdx:               [Optional] MultiIndex-ed Series of boolean array

    Returns
    -------
    Filtered MultiIndex-ed DataFrame, MultiIndex-ed Series of boolean array (True values match filtered indices)

    """
    # initialize boolArray[True] for selecting (booleanIndexing) blocks using criterion in ['selectionParams']
    if fIdx is None:
        fIdx = blocksinfo['MSO '] == blocksinfo['MSO ']

    # change the truth values of Index by doing string comparison on dataframe.Index
    epochIndices = blocksinfo.index.to_frame()
    for item in EPOCHISOLATORS:
        if (strings := analysis_params['selectionParams']['Epoch'][item]) is not None:
            if isinstance(strings, tuple):
                fIdx &= epochIndices[item].str.contains('|'.join(strings))
            else:
                fIdx &= epochIndices[item].str.contains(strings)

    # change the truth values of Index by doing floating point comparison on dataframe columns
    selectCols = analysis_params['selectionParams'].keys() & COLS_WITH_FLOATS
    for col in selectCols:
        string = analysis_params['selectionParams'][col]
        if re.match('<=', string):
            val = re.sub('<=', '', string)
            fIdx &= blocksinfo[col] <= np.float_(val)
        elif re.match('<', string):
            val = re.sub('<', '', string)
            fIdx &= blocksinfo[col] < np.float_(val)
        elif re.match('>=', string):
            val = re.sub('>=', '', string)
            fIdx &= blocksinfo[col] >= np.float_(val)
        elif re.match('>', string):
            val = re.sub('>', '', string)
            fIdx &= blocksinfo[col] > np.float_(val)
        elif re.match('==', string):
            val = re.sub('==', '', string)
            fIdx &= blocksinfo[col] == np.float_(val)

    # change the truth values of Index by doing string comparison on dataframe columns
    selectCols = analysis_params['selectionParams'].keys() & COLS_WITH_STRINGS
    for col in selectCols:
        strings = analysis_params['selectionParams'][col]
        if isinstance(strings, tuple):
            fIdx &= blocksinfo[col].str.contains('|'.join(strings))
        else:
            fIdx &= blocksinfo[col].str.contains(strings)

    return blocksinfo.loc[fIdx, :], fIdx


class FilterBlocks(object):
    """
    Compute filtered blocksinfo using analysis parameters and cache it
    """

    def __init__(self):
        self.cache = None

    def __get__(self, obj, objType):
        if self.cache is None:
            self.cache = _filter_blocks_helper(obj.blocksinfo, obj.analysis_params, fIdx=None)
        return self.cache[0], self.cache[1]

    def __set__(self, obj, value):
        self.cache = value

    def __call__(self, blocksinfo, analysis_params, filterIndices=None):
        return _filter_blocks_helper(blocksinfo, analysis_params, filterIndices)


class TMSTG(object):
    """TMSTG

    Enables easy access to data contained in h5py group object and importantly eases analysis on that data

    Parameters
    ----------
    matdata: {List} of MATdata objects that provide access to data on disk
    blocksinfo: {DataFrame} of read 'infofile'

    Note:   Do not instantiate this class directly. Instead, use...
            TMSTG.load(matlabfnames, ).
            @params: matlabfnames:  {Pandas} Series of full path filenames of .mat files
            @params: infofile:      full path filename of .xlsx "infofile"

    Attributes
    ---------
    psfr:                   {cached property}    get 'Peristimulus Firing Rate'
    psts:                   {cached property}    get 'Peristimulus time stamps'
    singleUnitsSpikeTimes:  {cached property}    get spikeTimes of single units, set 'index' for selection

    do_multi_indexing:      changes the index of "infofile" {DataFrame} to multiIndex

    """
    _default_analysis_params = {
        'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ]))},
        'TMSArtifactParams': {'timeWin': (-0.3, 0.5)},
        'peristimParams': {'smoothingParams': {'win': 'gaussian', 'width': 1.5, 'overlap': 1 / 3},
                           'timeWin': (-20.0, 100.0),
                           'trigger': 'TMS',
                           'baselinetimeWin': (-50.0, -1.0)}}

    analysis_params = th.AnalysisParams(_default_analysis_params)
    filter_blocks = FilterBlocks()
    late_comp = th.LateComponent()

    def __init__(self, matdata=None, blocksinfo=None) -> None:
        self.matdata: Optional['pd.Series[mp.MATdata]'] = matdata.sort_index()
        self.blocksinfo: Optional[pd.DataFrame] = blocksinfo.sort_index()

    @classmethod
    def load(cls, groupinfofilePath: str) -> 'TMSTG':
        """
        create a TMSTG object using a list of singleLocation files and an infofile

        Parameters
        ----------
        groupinfofilePath:   file location of groupinfofile

        Returns
        -------
        an instance of TMSTG class
        """

        groupinfo = pd.read_excel(groupinfofilePath).dropna()
        groupMatlabfnames = pd.Series()
        groupBlocksinfo = pd.DataFrame()

        for i in groupinfo.index:
            animalInfofilePath = [groupinfo.loc[i, 'Folder'] + '\\' + f
                                  for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.xlsx')]
            animalMatlabfnames = pd.Series(groupinfo.loc[i, 'Folder'] + '\\' + f
                                           for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.mat'))
            if len(animalInfofilePath) > 0:
                blocksinfo = pd.read_excel(animalInfofilePath[0]).drop(
                    columns=['Unnamed: 11', 'Queries'], errors='ignore').dropna(axis=0, how='all')
                th._concat_blocksinfo(blocksinfo, 'Animal', str(groupinfo.loc[i, 'Animal']))
                blocksinfo = cls.do_multi_indexing(blocksinfo, animalMatlabfnames)
                blocksinfo = th._edit_blocksinfo(blocksinfo, 'TrigIndices')
                cls._sort_filelist(animalMatlabfnames, blocksinfo)
                groupMatlabfnames = pd.concat([groupMatlabfnames, animalMatlabfnames], ignore_index=True)
                groupBlocksinfo = pd.concat([groupBlocksinfo, blocksinfo])

        return cls(
            pd.Series([mp.MATfile(fname).read() for fname in groupMatlabfnames], index=groupBlocksinfo.index.unique()),
            groupBlocksinfo)

    @cached_property
    def psts(self) -> list:
        """
        Compute peri-stimulus spike time-stamps

        Returns
        -------
        List containing spike time-stamps from each block

        """
        print('psts runs...........')
        th._check_trigger_numbers(self.matdata, self.blocksinfo)
        th._check_mso_order(self.matdata, self.blocksinfo)
        selectBlocks, selectBlocksIdx = self.filter_blocks

        psTS = list()
        for epochIndex, blockinfo in selectBlocks.iterrows():
            selectTrigger = th._get_trigger_times_for_current_block(
                self.analysis_params['peristimParams']['trigger'], self.matdata[epochIndex], blockinfo)
            timeIntervals = selectTrigger[:, np.newaxis] + np.array(self.analysis_params['peristimParams']['timeWin'])
            psTS.append(peristim_timestamp(self.singleUnitsSpikeTimes(epochIndex), timeIntervals))

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
        Compute peri-stimulus firing rate

        Returns
        -------
        tuple having list of peri-stimulus firing rate and a 1-D array of its timing
        """

        print('compute_firingrate runs...........')
        th._check_trigger_numbers(self.matdata, self.blocksinfo)
        th._check_mso_order(self.matdata, self.blocksinfo)
        selectBlocks, selectBlocksIdx = self.filter_blocks

        ps_FR, ps_T = list(), np.array([])
        for epochIndex, blockinfo in selectBlocks.iterrows():
            selectTrigger = th._get_trigger_times_for_current_block(triggerType,
                                                                    self.matdata[epochIndex], blockinfo)
            timeIntervals = selectTrigger[:, np.newaxis] + np.array([timeWinLeftEndpoint, timeWinRightEndpoint])
            tmp_ps_FR, ps_T = peristim_firingrate(self.singleUnitsSpikeTimes(epochIndex),
                                                  timeIntervals,
                                                  {'win': smoothingWinType,
                                                   'width': smoothingWinWidth,
                                                   'overlap': smoothingWinOverlap})
            ps_FR.append(tmp_ps_FR)

        return ps_FR, ps_T

    @staticmethod
    def do_multi_indexing(blocksinfo: pd.DataFrame, matlabfnames: pd.Series) -> pd.DataFrame:
        """
        Multi-indexes the passed in DataFrame

        Parameters
        ----------
        blocksinfo: [DataFrame] read infofile (having block information).

        Returns
        -------
        MultiIndex-ed DataFrame containing block information.
        """
        df = blocksinfo.copy()
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('µm')
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('μm')
        df['Depth_int'] = np.int_(df.loc[:, 'Depth'].to_list())
        df['no. of Trigs'] = np.int_(df.loc[:, 'no. of Trigs'].to_list())

        for key in REGIONS:
            df.loc[df['RecArea '].str.contains('|'.join(REGIONS[key])), EPOCHISOLATORS[1]] = key
        recRegions = df.loc[:, EPOCHISOLATORS[1]].unique()

        df.loc[:, EPOCHISOLATORS[2]] = np.nan
        corticalRegions = set(recRegions) & set(LAYERS.keys())
        for cRegion in corticalRegions:
            for key in LAYERS[cRegion]:
                df.loc[(LAYERS[cRegion][key][0] <= df['Depth_int'])
                       & (df['Depth_int'] < LAYERS[cRegion][key][1])
                       & df['Region'].str.contains(cRegion), EPOCHISOLATORS[2]] \
                    = key

        if not df['StimHem'].isna().all():
            df.loc[df['StimHem'] == df['RecHem'], EPOCHISOLATORS[3]] = 'same'
            df.loc[df['StimHem'] != df['RecHem'], EPOCHISOLATORS[3]] = 'opposite'
        else:
            df.loc[:, EPOCHISOLATORS[3]] = np.nan

        if all(map(lambda x: isinstance(x, str), df['Filename'])):
            df.loc[df['Filename'].str.contains('con'), EPOCHISOLATORS[4]] = 'contra'
            df.loc[df['Filename'].str.contains('ips'), EPOCHISOLATORS[4]] = 'ipsi'
        elif df['Filename'].isna().all():
            df.loc[:, EPOCHISOLATORS[4]] = np.nan
        else:
            raise ValueError(f'The infofile column \'Filename\' cannot be half empty '
                             f'which is the case for animal {df["Animal"][0]}')

        if not (matlabfnames.str.contains('|'.join(('same', 'opposite')))
                | matlabfnames.str.contains('|'.join(('con', 'ips')))).any():
            df.loc[:, 'Movement'] = df.loc[:, 'Mov'].copy()
            df.loc[:, EPOCHISOLATORS[3:5]] = np.nan

        df.fillna('none', inplace=True)
        df.set_index(EPOCHISOLATORS, inplace=True)
        df.sort_index(inplace=True)

        return df

    def avg_FR_per_neuron(self, squeezeDim=True) -> tuple[np.ndarray | list, np.ndarray, np.ndarray | list, np.ndarray]:
        """
        Calculate average peristimulus firing rate of each neuron

        Parameters
        ----------
        squeezeDim: [Flag]  True:   returns 2-D array
                            False:  returns 3-D array (mean computed over axis=0)

        Returns
        -------
        array[Time X N] - average peristimulus activity,
        array[1D]       - Time points of peristimulus activity,
        array[1 X N]    - baseline firing rate,
        array[1D]       - mid-time point of baseline firing rate

        or

        list of arrays[[1 X Time X N], ]    - average peristimulus activity,
        array[1D]                           - Time points of peristimulus activity,
        list of arrays[[1 X N], ]           - baseline firing rate,
        array[1D]                           - mid-time point of baseline firing rate
        """
        # noinspection DuplicatedCode
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
                ps_T + self.analysis_params['peristimParams']['timeWin'][0], \
                np.concatenate([block_bsfr.mean(axis=0) for block_bsfr in ps_baseline_FR], axis=1), \
                ps_baseline_T + mean(self.analysis_params['peristimParams']['baselinetimeWin'])
        else:
            return [block_psfr.mean(axis=0, keepdims=True) for block_psfr in ps_FR], \
                ps_T + self.analysis_params['peristimParams']['timeWin'][0], \
                [block_bsfr.mean(axis=0, keepdims=True) for block_bsfr in ps_baseline_FR], \
                ps_baseline_T + mean(self.analysis_params['peristimParams']['baselinetimeWin'])

    def stats_is_signf_active(self,
                              ps_FR: Optional[list[np.ndarray]] = None,
                              ps_baseline_FR: Optional[np.ndarray] = None) -> list[np.ndarray[bool]]:

        if ps_FR is None and ps_baseline_FR is None:
            ps_FR, _ = self.compute_firingrate('rectangular',
                                               49.0,
                                               0.0,
                                               25.5,
                                               50.0,
                                               self.analysis_params['peristimParams']['trigger'])
            ps_baseline_FR, _ = self.compute_firingrate('rectangular',
                                                        49.0,
                                                        0.0,
                                                        -25.5,
                                                        -1,
                                                        self.analysis_params['peristimParams']['trigger'])
            selectBlocksinfo, selectBlocksinfoIdx = self.filter_blocks
            mtCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}}
            fb = FilterBlocks()
            _, mtIdx = fb(selectBlocksinfo, mtCond)

            epochs = selectBlocksinfo.index.unique()
            activeNeu = pd.Series(np.empty(shape=epochs.shape), index=epochs)
            for epoch in epochs:
                indices = np.where((selectBlocksinfo.index == epoch) & mtIdx.to_numpy())[0]
                if len(indices) > 0:
                    postStimFR = ps_FR[indices[0]]
                    baselineFR = ps_baseline_FR[indices[0]]
                    for index in indices[1:]:
                        postStimFR = np.append(postStimFR, ps_FR[index], axis=0)
                        baselineFR = np.append(baselineFR, ps_baseline_FR[index], axis=0)
                    activeNeu[epoch] = scipy.stats.ttest_ind(postStimFR.reshape(postStimFR.shape[0::2]),
                                                             baselineFR.reshape(baselineFR.shape[0::2]),
                                                             axis=0,
                                                             alternative='greater').pvalue < 0.05
                else:
                    indices = np.where(selectBlocksinfo.index == epoch)[0]
                    activeNeu[epoch] = np.repeat(False, ps_FR[indices[0]].shape[2], axis=0)

        return activeNeu

    @lru_cache(maxsize=None)
    def singleUnitsSpikeTimes(self, epochIndex) -> 'nb.typed.List':
        multiUnitSpikeTimes: np.ndarray = self.matdata.loc[epochIndex]['SpikeModel/SpikeTimes/data'].flatten()
        refs = self.matdata.loc[epochIndex]['SpikeModel/ClusterAssignment/data'].flatten()
        singleUnitsIndices = [self.matdata.loc[epochIndex][i].flatten().astype(int) - 1 for i in refs]
        if self.analysis_params['TMSArtifactParams'] is not None:
            singleUnitsIndices \
                = self._remove_spikes_within_TMSArtifact_timeWin(multiUnitSpikeTimes, singleUnitsIndices, epochIndex)
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

        def lookfor_matching_fname(boolArray, *string) -> pd.Series:

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
        epochIndices = blocksinfo.index.unique().to_numpy()
        for i, epochIndex in enumerate(epochIndices):
            matchBoolArray = lookfor_matching_fname(np.ones(len(matlabfnames), dtype=bool), *epochIndex)
            j = matchBoolArray.array.argmax()
            matlabfnames.iloc[[i, j]] = matlabfnames.iloc[[j, i]]

        pass

    def _remove_spikes_within_TMSArtifact_timeWin(self,
                                                  spikeTimes: np.ndarray,
                                                  singleUnitsIndices: list,
                                                  epochIndex: tuple) -> list[np.ndarray]:
        """
        Removes spikes that fall inside the closed interval specified in 'TMSArtifactParams'

        Parameters
        ----------
        spikeTimes:             multiUnitSpikeTimes
        singleUnitsIndices:     list of single unit indices
        epochIndex:             index of matdata from which multiUnitSpikeTimes were extracted

        Returns
        -------
        list of single unit indices that fall outside the closed interval specified in 'TMSArtifactParams'
        """

        refs = self.matdata[epochIndex]['rawData/trigger'].flatten()
        trigChanIdx = self.matdata[epochIndex]['TrigChan_ind'][0, 0].astype(int) - 1
        ampTrigIdx = 2

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
            print(f'Amplifier trigger channel no. {ampTrigIdx} is empty for epoch {epochIndex},... ')
            print(f'therefore using trigger channel no. {trigChanIdx} to remove spikes around TMSArtifact....')
            print(f'As the width of the trigger for this channel is {np.mean(trigger[1::2] - trigger[::2])} secs, ')
            print(f'it is expanded to match amplifier trig width of {1 + np.mean(trigger[1::2] - trigger[::2])} secs.')
            print('')
            timeIntervals = (trigger.reshape((trigger.size // 2, 2))
                             + (np.array([-0.2, 0.8], dtype=trigger.dtype)
                                + np.array(self.analysis_params['TMSArtifactParams']['timeWin'])))

        mask = np.zeros_like(spikeTimes, dtype=bool)
        for timeInterval in timeIntervals:
            mask |= (timeInterval[0] <= spikeTimes) & (spikeTimes <= timeInterval[1])
        mask = mask.nonzero()

        return [np.setdiff1d(sU_indices, mask) for sU_indices in singleUnitsIndices]


if __name__ == '__main__':
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)

    th._check_trigger_numbers(tms.matdata, tms.blocksinfo)
    th._check_mso_order(tms.matdata, tms.blocksinfo)

    tms.analysis_params = {'selectionParams': {'Epoch': {'Region': 'MC',
                                                         'Layer': 'L5'},
                                               'MT': '>=0.9'}}

    # tms.analysis_params = {'selectionParams': {'Epoch': {'Region': 'thal'},
    #                                            'MT': '>=1.2',
    #                                            'RecArea ': ('VPM', 'PO', 'VM', 'VPL')}}

    meanPSFR, t, meanBaselineFR, _ = tms.avg_FR_per_neuron()
    activeNeu = tms.stats_is_signf_active()
    selectBlocks, selectBlocksIdx = tms.filter_blocks
    flat_activeNeu = np.empty(shape=0, dtype=bool)
    [flat_activeNeu := np.append(flat_activeNeu, activeNeu[item]) for item in selectBlocks.index]
    delays = tms.late_comp.compute_delay(meanPSFR[:, flat_activeNeu],
                                         t,
                                         meanPSFR[t < 0, :].max(axis=0, keepdims=True)[:, flat_activeNeu],
                                         tms.analysis_params['peristimParams']['smoothingParams']['width'] + 0.25)
    delays
