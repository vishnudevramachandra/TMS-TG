import numpy as np
import numba as nb
import pandas as pd
import lib.matpy as mp
import os
import re
from typing import Optional
from itertools import zip_longest
from functools import cached_property
import lib.helper_tms_tg as th

LAYERS = {'L23': (200, 800), 'L4': (800, 1200), 'L5': (1200, 1600), 'L6': (1600, 1900)}
REGIONS = {'thal': ['BZ', 'CZ'], 'BG': ['STN'], 'MC': ['CFA', 'MC'], 'SC': ['S1', 'SC'], 'VC': ['V1', 'VC']}
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
    _default_analysis_params = {'selectionParams': {
        'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ]))},
        'smoothingParams': {'win': 'gauss', 'width': 2.0, 'overlap': 1 / 2},
        'timeWin': (-20, 100),
        'trigger': {'TMS': None},
        'baselinetimeWin': (-50, -1)}

    analysis_params = th.AnalysisParams(_default_analysis_params)
    psfr = th.PSFR()
    # psts = th.Raster()

    # TODO: LateComponent implementation
    late_comp = th.LateComponent()

    def __init__(self, matdata=None, epochinfo=None) -> None:
        self.matdata: Optional[pd.Series[mp.MATdata]] = matdata
        self.epochinfo: Optional[pd.DataFrame] = epochinfo
        self._filter_blocks

    @classmethod
    def load(cls, path: str) -> 'TMSTG':
        """
        create a TMSTG object using a list of singleLocation files and an infofile
        """
        groupinfofilePath = [path + '\\' + f for f in os.listdir(path) if f.endswith('.xlsx')]

        if len(groupinfofilePath) > 0:

            groupinfo = pd.read_excel(groupinfofilePath[0]).dropna()
            groupMatlabfnames: 'pd.Series[str]' = pd.Series()
            groupEpochinfo: 'pd.DataFrame' = pd.DataFrame()

            for i in groupinfo.index:
                animalInfofilePath = [groupinfo.loc[i, 'Folder'] + '\\' + f
                                      for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.xlsx')]
                animalMatlabfnames = pd.Series(groupinfo.loc[i, 'Folder'] + '\\' + f
                                               for f in os.listdir(groupinfo.loc[i, 'Folder']) if f.endswith('.mat'))
                if len(animalInfofilePath) > 1:
                    epochinfo = pd.read_excel(animalInfofilePath[0]).dropna()
                    epochinfo['Animal'] = str(groupinfo.loc[i, 'Animal'])
                    epochinfo = cls.do_multi_indexing(epochinfo)
                    cls._sort_filelist(animalMatlabfnames, epochinfo)
                    groupMatlabfnames = pd.concat([groupMatlabfnames, animalMatlabfnames], ignore_index=True)
                    groupEpochinfo = pd.concat([groupEpochinfo, epochinfo])

            return cls(
                pd.Series([mp.MATfile(fname).read() for fname in groupMatlabfnames],
                          index=groupEpochinfo.index.unique()),
                groupEpochinfo)

        else:

            return cls()

    @staticmethod
    def do_multi_indexing(epochinfo: pd.DataFrame) -> pd.DataFrame:
        """
        sorts the order of matlabf<ile>names in the list to be consistent with epoch order
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

        df.loc[df['StimHem'] == df['RecHem'], EPOCHISOLATORS[3]] = 'same'
        df.loc[df['StimHem'] != df['RecHem'], EPOCHISOLATORS[3]] = 'opposite'
        df.fillna('missing', inplace=True)

        df.loc[df['Filename'].str.contains('con'), EPOCHISOLATORS[4]] = 'contra'
        df.loc[df['Filename'].str.contains('ips'), EPOCHISOLATORS[4]] = 'ipsi'

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

    @cached_property
    def singleUnitsSpikeTimes(self):
        uniqueEpochs = self.epochinfo.index.unique().to_numpy()
        output = dict()
        for uniqueEpoch in uniqueEpochs:
            output[uniqueEpoch] = self._singleUnitsSpikeTimes_loop(uniqueEpoch)

        return output

    def _singleUnitsSpikeTimes_loop(self, index):
        multiUnitSpikeTimes: np.ndarray = self.matdata.loc[index]['SpikeModel/SpikeTimes/data'].flatten()
        refs = self.matdata.loc[index]['SpikeModel/ClusterAssignment/data'].flatten()
        spikeIndices_of_neurons = [self.matdata.loc[index][i].flatten().astype(int) - 1 for i in refs]
        singleUnitsSpikeTimes = nb.typed.List()
        [singleUnitsSpikeTimes.append(multiUnitSpikeTimes[indices]) for indices in spikeIndices_of_neurons]
        return singleUnitsSpikeTimes

    @staticmethod
    def _sort_filelist(matlabfnames, epochinfo) -> None:
        """
        sorts the order of matlabf<ile>names in the list to be consistent with epoch order
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
        uniqueEpochs = pd.unique(epochinfo.index)
        for i, uniqueEpoch in enumerate(uniqueEpochs):
            boolIndex = lookfor_matching_fname(np.ones(len(matlabfnames), dtype=bool), *uniqueEpoch)
            j = boolIndex.array.argmax()
            matlabfnames.iloc[[i, j]] = matlabfnames.iloc[[j, i]]

        pass

    @cached_property
    def _filter_blocks(self):
        # initialize boolArray[True] for selecting (booleanIndexing) blocks using criterion in ['selectionParams']
        idx = self.epochinfo['MSO '] == self.epochinfo['MSO ']

        # change the truth values of Index by doing string comparison on dataframe.Index
        epochIndex = self.epochinfo.index.to_frame()
        for item in EPOCHISOLATORS:
            if (strings := self.analysis_params['selectionParams']['Epoch'][item]) is not None:
                idx &= epochIndex[item].str.contains('|'.join(strings))

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


if __name__ == '__main__':

    animalListFolder = r'G:\Vishnu\data\TMSTG'
    tms = TMSTG.load(animalListFolder)
    # tms.psfr

    # dir_path = r'G:\Vishnu\data\TMSTG\20180922'
    # matlabfiles = pd.Series(dir_path + '\\' + f for f in os.listdir(dir_path) if f.endswith('.mat'))
    # infofile = [dir_path + '\\' + f for f in os.listdir(dir_path) if f.endswith('.xlsx')]
    # tms = TMSTG.load(matlabfiles, infofile[0])

    tms.analysis_params = {'selectionParams': {'Epoch': {'Region': 'MC'}, 'MT': '>=1'}}
    tms.psfr
    tms.psfr
