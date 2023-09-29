import numpy as np
import numba as nb
import pandas as pd
import lib.matpy as mp
import os
import re
from typing import Optional, Any
from itertools import zip_longest
from functools import cached_property, lru_cache
import lib.helper_tms_tg as th
from scipy import stats

LAYERS = {'L23': (200, 800), 'L4': (800, 1200), 'L5': (1200, 1600), 'L6': (1600, 1900)}
REGIONS = {'thal': ['BZ', 'CZ', 'CL', 'PC'],
           'BG': ['STN'],
           'MC': ['CFA', 'MC'],
           'SC': ['S1', 'SC'],
           'VC': ['V1', 'VC']}
COILDIR = {'ML', 'LM', 'PA', 'AP'}
STIMHEM = {'LH', 'RH'}
COILPOS = {'MC', 'SC', 'VC'}
MANIPULATION = {'MUS'}

EPOCHISOLATORS = ['Animal', 'Region', 'Layer', 'CoilHemVsRecHem', 'Mov', 'Depth']
COLS_WITH_FLOATS = {'MSO ', 'MT', 'no. of Trigs', 'Stimpulses', 'Depth_int'}
COLS_WITH_STRINGS = {'StimHem', 'CoilDir', 'TG-Injection ', 'RecArea ', 'RecHem', 'Filename'}


class TMSTG(object):
    """TMSTG

    Enables easy access to data contained in h5py group object

    Parameters
    ----------
    matdata: {List} of MATdata objects that provide access to data on disk
    epochinfo: {DataFrame} of read 'infofile'

    Note:   Do not instantiate this class directly. Instead, use...
            TMSTG.load(matlabfnames, ).
            @params: matlabfnames:  {Pandas} Series of full path filenames of .mat files
            @params: infofile:      full path filename of .xlsx "infofile"

    Attributes
    ---------
    psfr:       {Descriptor}    get 'Peristimulus Firing Rate' and set 'parameters' for computing it
    spikeTimes: {Descriptor}    get spikeTimes of single units, set 'index' for selection
    lateComp:

    do_multi_indexing:      changes the index of "infofile" {DataFrame} to multiIndex


    """
    _default_analysis_params = {
        'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ]))},
        'TMSArtifactParams': {'timeWin': (-0.3, 0.3)},
        'peristimParams': {'smoothingParams': {'win': 'gauss', 'width': 2.0, 'overlap': 1 / 2},
                           'timeWin': (-20, 100),
                           'trigger': {'TMS': None},
                           'baselinetimeWin': (-50, -1)},
        'lateComponentParams': {'minDelay': 10, 'method': ('std', 3)}}

    analysis_params = th.AnalysisParams(_default_analysis_params)
    psfr = th.PSFR()
    psts = th.Raster()

    # TODO: LateComponent implementation
    late_comp = th.LateComponent()

    def __init__(self, matdata=None, epochinfo=None) -> None:
        self.matdata: Optional['pd.Series[mp.MATdata]'] = matdata.sort_index()
        self.epochinfo: Optional[pd.DataFrame] = epochinfo.sort_index()

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
        groupEpochinfo = pd.DataFrame()

        for i in groupinfo.index:
            animalInfofilePath = [groupinfo.loc[i, 'Folder'] + '\\' + f
                                  for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.xlsx')]
            animalMatlabfnames = pd.Series(groupinfo.loc[i, 'Folder'] + '\\' + f
                                           for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.mat'))
            if len(animalInfofilePath) > 0:
                epochinfo = pd.read_excel(animalInfofilePath[0]).drop(
                    columns=['Unnamed: 11', 'Queries'], errors='ignore').dropna(axis=0, how='all')
                cls._concatEpochinfo(epochinfo, 'Animal', str(groupinfo.loc[i, 'Animal']))
                epochinfo = cls.do_multi_indexing(epochinfo)
                cls._concatEpochinfo(epochinfo, 'TrigStartIdx')
                cls._sort_filelist(animalMatlabfnames, epochinfo)
                groupMatlabfnames = pd.concat([groupMatlabfnames, animalMatlabfnames], ignore_index=True)
                groupEpochinfo = pd.concat([groupEpochinfo, epochinfo])

        return cls(
            pd.Series([mp.MATfile(fname).read() for fname in groupMatlabfnames], index=groupEpochinfo.index.unique()),
            groupEpochinfo)

    @staticmethod
    def _concatEpochinfo(epochinfo: pd.DataFrame, colName: str, value: Optional[Any] = None) -> None:
        """
        Concatenates a new column to passed in DataFrame

        Parameters
        ----------
        epochinfo:  DataFrame containing epoch information. If no argument is passed to
                    'value' parameter then the DataFrame has to be multi-indexed
        colName:    name for new column
        value:      [Optional] value that is added to all rows of DataFrame

        """

        if colName not in epochinfo.columns:

            if value is not None:
                epochinfo[colName] = value

            else:
                match colName:
                    case 'TrigStartIdx':
                        epochinfo['TrigStartIdx'] = 0
                        epochIndices = epochinfo.index.unique().to_numpy()
                        for epochIndex in epochIndices:
                            num_of_trigs = epochinfo.loc[epochIndex, 'no. of Trigs'].to_numpy()
                            epochinfo.loc[epochIndex, 'TrigStartIdx'] = np.append(0, num_of_trigs.cumsum()[:-1])
                    case _:
                        print(f'Not implemented : Adding column with title "{colName}" without '
                              f'a given value')

    @staticmethod
    def do_multi_indexing(epochinfo: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-indexes the passed in DataFrame

        Parameters
        ----------
        epochinfo:  DataFrame containing epoch information.

        """
        df = epochinfo.copy()
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('µm')
        df['Depth_int'] = np.int_(df.loc[:, 'Depth'].to_list())
        df['no. of Trigs'] = np.int_(df.loc[:, 'no. of Trigs'].to_list())

        for key in REGIONS:
            df.loc[df['RecArea '].str.contains('|'.join(REGIONS[key])), EPOCHISOLATORS[1]] = key

        for key in LAYERS:
            df.loc[(LAYERS[key][0] <= df['Depth_int'])
                   & (df['Depth_int'] < LAYERS[key][1]), EPOCHISOLATORS[2]] \
                = key
        if not df['StimHem'].isna().all():
            df.loc[df['StimHem'] == df['RecHem'], EPOCHISOLATORS[3]] = 'same'
            df.loc[df['StimHem'] != df['RecHem'], EPOCHISOLATORS[3]] = 'opposite'
        else:
            df.loc[:, EPOCHISOLATORS[3]] = np.nan

        df.loc[df['Filename'].str.contains('con'), EPOCHISOLATORS[4]] = 'contra'
        df.loc[df['Filename'].str.contains('ips'), EPOCHISOLATORS[4]] = 'ipsi'

        df.fillna('none', inplace=True)
        df.set_index(EPOCHISOLATORS, inplace=True)
        df.sort_index(inplace=True)
        return df

    def avg_FR_per_neuron(self, squeezeDim=True):
        """
        Calculate average peristimulus firing rate

        Parameters
        ----------
        squeezeDim: [Flag]  True:   returns matrix
                            False:  collapsed over row axis

        Returns
        -------
        matrix[N X Time] or list of matrices[[N X Time]]

        or

        matrix[1 X Time X N] or list of matrices[[1 X Time X N]]
        """

        ps_FR, ps_T, ps_baseline_FR, ps_baseline_T = self.psfr

        if squeezeDim:
            return np.concatenate([block_psfr.mean(axis=0) for block_psfr in ps_FR], axis=1).T, \
                   ps_T, \
                   np.concatenate([block_bsfr.mean(axis=0) for block_bsfr in ps_baseline_FR], axis=1).T, \
                   ps_baseline_T
        else:
            return [block_psfr.mean(axis=0, keepdims=True) for block_psfr in ps_FR], \
                   ps_T, \
                   [block_bsfr.mean(axis=0, keepdims=True) for block_bsfr in ps_baseline_FR], \
                   ps_baseline_T

    @lru_cache(maxsize=None)
    def singleUnitsSpikeTimes(self, epochIndex) -> 'nb.typed.List':
        multiUnitSpikeTimes: np.ndarray = self.matdata.loc[epochIndex]['SpikeModel/SpikeTimes/data'].flatten()
        refs = self.matdata.loc[epochIndex]['SpikeModel/ClusterAssignment/data'].flatten()
        singleUnitsIndices = [self.matdata.loc[epochIndex][i].flatten().astype(int) - 1 for i in refs]
        if (self.analysis_params['TMSArtifactParams'] is not None
                or self.analysis_params['TMSArtifactParams'] is not None):
            singleUnitsIndices \
                = self._remove_spikes_within_TMSArtifact_timeWin(multiUnitSpikeTimes, singleUnitsIndices, epochIndex)
        singleUnitsSpikeTimes = nb.typed.List()
        [singleUnitsSpikeTimes.append(multiUnitSpikeTimes[sU_indices]) for sU_indices in singleUnitsIndices]
        return singleUnitsSpikeTimes

    @staticmethod
    def _sort_filelist(matlabfnames, epochinfo) -> None:
        """
        Sorts the order of matlabf<ile>names in the Series to be consistent with epoch order

        Parameters
        ----------
        matlabfnames:   Series of matlabf<ile>names
        epochinfo:      MultiIndex-ed DataFrame containing epoch information

        """

        def lookfor_matching_fname(boolArray, *string) -> pd.Series:

            if (boolArray & matlabfnames.str.contains(string[0])).any() & (len(string) > 1):
                boolArray &= matlabfnames.str.contains(string[0])
                return lookfor_matching_fname(boolArray, *string[1:])
            elif (not (boolArray & matlabfnames.str.contains(string[0])).any()) & (len(string) > 1):
                return lookfor_matching_fname(boolArray, *string[1:])
            elif (boolArray & matlabfnames.str.contains(string[0])).any() & (len(string) == 1):
                return boolArray & matlabfnames.str.contains(string[0])
            else:
                return boolArray

        # Use MultiIndexes of epochinfo to set the file order
        epochIndices = epochinfo.index.unique().to_numpy()
        for i, epochIndex in enumerate(epochIndices):
            boolIndex = lookfor_matching_fname(np.ones(len(matlabfnames), dtype=bool), *epochIndex)
            j = boolIndex.array.argmax()
            matlabfnames.iloc[[i, j]] = matlabfnames.iloc[[j, i]]

        pass

    @cached_property
    def filter_blocks(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Filters the epoch MultiIndex-ed DataFrame using 'selectionParams' stored in self.analysis_params

        Returns
        -------
        Filtered MultiIndex-ed DataFrame, MultiIndex-ed Series of boolean array (True values match filtered indices)

        """

        # initialize boolArray[True] for selecting (booleanIndexing) blocks using criterion in ['selectionParams']
        idx = self.epochinfo['MSO '] == self.epochinfo['MSO ']

        # change the truth values of Index by doing string comparison on dataframe.Index
        epochIndices = self.epochinfo.index.to_frame()
        for item in EPOCHISOLATORS:
            if (strings := self.analysis_params['selectionParams']['Epoch'][item]) is not None:
                idx &= epochIndices[item].str.contains('|'.join(strings))

        # change the truth values of Index by doing floating point comparison on dataframe columns
        selectCols = self.analysis_params['selectionParams'].keys() & COLS_WITH_FLOATS
        for col in selectCols:
            string = self.analysis_params['selectionParams'][col]
            if re.match('<=', string):
                val = re.sub('<=', '', string)
                idx &= self.epochinfo[col] <= np.float_(val)
            elif re.match('<', string):
                val = re.sub('<', '', string)
                idx &= self.epochinfo[col] <= np.float_(val)
            elif re.match('>=', string):
                val = re.sub('>=', '', string)
                idx &= self.epochinfo[col] >= np.float_(val)
            elif re.match('>', string):
                val = re.sub('>', '', string)
                idx &= self.epochinfo[col] > np.float_(val)
            elif re.match('==', string):
                val = re.sub('==', '', string)
                idx &= self.epochinfo[col] == np.float_(val)

        # change the truth values of Index by doing string comparison on dataframe columns
        selectCols = self.analysis_params['selectionParams'].keys() & COLS_WITH_STRINGS
        for col in selectCols:
            string = self.analysis_params['selectionParams'][col]
            idx &= self.epochinfo[col].str.contains(string)

        return self.epochinfo.loc[idx, :], idx

    def _remove_spikes_within_TMSArtifact_timeWin(self,
                                                  spikeTimes: np.ndarray,
                                                  singleUnitsIndices: list,
                                                  epochIndex: tuple) -> np.ndarray:
        """
        Removes spikes that fall inside the closed interval specified in 'TMSArtifactParams'

        Parameters
        ----------
        spikeTimes:     multiUnitSpikeTimes
        epochIndex:     index of matdata from which multiUniSpikeTimes were extracted

        Returns
        -------
        multiUnitSpikeTimes with spikes removed

        """
        refs = self.matdata[epochIndex]['rawData/trigger'].flatten()
        trigChanIdx = self.matdata[epochIndex]['TrigChan_ind'][0, 0].astype(int) - 1
        ampTrigIdx = 2

        if len(ampTrigger := self.matdata[epochIndex][refs[ampTrigIdx]].flatten() * 1e3) \
                >= len(trigger := self.matdata[epochIndex][refs[trigChanIdx]].flatten() * 1e3):
            if len(ampTrigger) > len(trigger):
                # take care of wrong ampTriggers
                ampTrigger = ampTrigger.reshape((ampTrigger.size // 2, 2))
                cond = stats.mode(np.diff(ampTrigger, axis=1))[0]
                i = 0
                while i < ampTrigger.shape[0]:
                    if (ampTrigger[i, 1] - ampTrigger[i, 0]) < (cond - 0.1):
                        ampTrigger = np.delete(ampTrigger, i * 2 + np.array([1, 2]))
                        ampTrigger = ampTrigger.reshape(ampTrigger.size // 2, 2)
                    i += 1
                assert ampTrigger.size == len(trigger), \
                    f'for epoch {epochIndex} length of ampTrigger and tmsTrigger are not the same'

            timeIntervals = ampTrigger.reshape((ampTrigger.size // 2, 2)) \
                            + np.array(self.analysis_params['TMSArtifactParams']['timeWin'])

        else:
            timeIntervals = trigger.reshape((trigger.size // 2, 2)) \
                            + np.array(self.analysis_params['TMSArtifactParams']['timeWin'])
            print(f'amplifier trigger channel no. {ampTrigIdx} is empty for epoch {epochIndex},... ')
            print(f'therefore using trigger channel no. {trigChanIdx} to remove spikes around TMSArtifact....')
            print(f'width of the trigger for this channel is {np.mean(trigger[1::2] - trigger[::2])} secs ')

        mask = np.zeros_like(spikeTimes, dtype=bool)
        for timeInterval in timeIntervals:
            mask |= (timeInterval[0] <= spikeTimes) & (spikeTimes <= timeInterval[1])
        mask = mask.nonzero()

        return [np.setdiff1d(sU_indices, mask) for sU_indices in singleUnitsIndices]


if __name__ == '__main__':
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    # tms.psfr

    # dir_path = r'G:\Vishnu\data\TMSTG\20180922'
    # matlabfiles = pd.Series(dir_path + '\\' + f for f in os.listdir(dir_path) if f.endswith('.mat'))
    # infofile = [dir_path + '\\' + f for f in os.listdir(dir_path) if f.endswith('.xlsx')]
    # tms = TMSTG.load(matlabfiles, infofile[0])
    th._check_trigger_numbers(tms.matdata, tms.epochinfo)
    th._check_mso_order(tms.matdata, tms.epochinfo)

    tms.analysis_params = {'selectionParams': {'Epoch': {'Region': 'MC',
                                                         'Layer': 'L23'},
                                               'MT': '>1'}}
    ps_TS = tms.psts
    ps_FR, ps_T, ps_baselineFR, _ = tms.psfr
    tms.psfr
