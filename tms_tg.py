import numpy as np
import pandas as pd
import lib.matpy as mp
import os
from typing import Optional
from itertools import zip_longest
from lib.helper_tms_tg import PSFR, LateComponent


LAYERS = {'L23': (200, 800), 'L4': (800, 1200), 'L5': (1200, 1600), 'L6': (1600, 1900)}
REGIONS = {'thal': ['BZ', 'CZ'], 'BG': ['STN'], 'MC': ['CFA', 'MC'], 'SC': ['S1', 'SC'], 'VC': ['V1', 'VC']}
COILDIR = {'ML', 'LM', 'PA', 'AP'}
STIMHEM = {'LH', 'RH'}
COILPOS = {'MC', 'SC', 'VC'}
MANIPULATION = {'MUS'}

EPOCHISOLATORS = ['Region', 'Layer', 'CoilHemVsRecHem', 'Mov', 'Depth']

class TMSTG(object):
    """TMSTG

    Enables easy access to data contained in h5py group object

    Parameters
    ----------
    matdata: {List} of MATdata objects that provide access to data on disk
    epochinfo: {DataFrame} containing read 'infofile' data

    Note:   Do not instantiate this class directly. Instead, use...
            TMSTG.load(matlabfnames, ).
            @params: matlabfnames:  {Pandas} Series of full path filenames of .mat files
            @params: infofile:      full path filename of .xlsx "infofile"

    Attributes
    ---------
    psfr:                   call 'Peristimulus Firing Rate' and set 'parameters' for computing it

    lateComp:

    do_multi_indexing:      changes the index of "infofile" {DataFrame} to multiIndex


    """
    psfr_default_params = {'selectionParams': {
                               'Epoch': dict(zip_longest(EPOCHISOLATORS, [None]))},
                           'smoothingParams': {'win': 'gauss', 'width': 2.0, 'overlap': 1 / 2},
                           'timeWin': (-20, 100),
                           'trigger': {'TMS': None},
                           'baselinetimeWin': (-50, -1)}

    psfr = PSFR()

    # TODO: LateComponent implimentation
    late_comp = LateComponent()

    def __init__(self, matdata=None, epochinfo=None) -> None:

        self.matdata: list[mp.MATdata] = matdata if matdata is not None else list()
        self.epochinfo: Optional[pd.DataFrame] = epochinfo

    @classmethod
    def load(cls, matlabfnames: 'pd.Series[str]', infofile: Optional[str] = None) -> 'TMSTG':
        """
        create a TMSTG object using a list of singleLocation files and an infofile
        """

        if infofile is not None:
            epochinfo = pd.read_excel(infofile).dropna()
            epochinfo = cls.do_multi_indexing(epochinfo)
            _sort_filelist(matlabfnames, epochinfo)
        else:
            epochinfo = None

        return cls([mp.MATfile(fname).read() for fname in matlabfnames], epochinfo)

    @staticmethod
    def do_multi_indexing(epochinfo: pd.DataFrame) -> pd.DataFrame:
        """
        sorts the order of matlabf<ile>names in the list to be consistent with epoch order
        """

        df = epochinfo.copy()
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('µm')
        df['Depth_int'] = np.int_(df.loc[:, 'Depth'].to_list())

        for key in REGIONS:
            df.loc[df['RecArea '].str.contains('|'.join(REGIONS[key])), EPOCHISOLATORS[0]] = key

        for key in LAYERS:
            df.loc[(LAYERS[key][0] <= df['Depth_int'])
                   & (df['Depth_int'] < LAYERS[key][1]), EPOCHISOLATORS[1]] \
                = key

        df.loc[df['StimHem'] == df['RecHem'], EPOCHISOLATORS[2]] = 'same'
        df.loc[df['StimHem'] != df['RecHem'], EPOCHISOLATORS[2]] = 'opposite'
        df.fillna('missing', inplace=True)

        df.loc[df['Filename'].str.contains('con'), EPOCHISOLATORS[3]] = 'contra'
        df.loc[df['Filename'].str.contains('ips'), EPOCHISOLATORS[3]] = 'ipsi'

        df.set_index(EPOCHISOLATORS, inplace=True)
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
            return np.concatenate([block_psfr.mean(axis=0) for block_psfr in ps_FR], axis=1).T,\
                   ps_T,\
                   np.concatenate([block_bsfr.mean(axis=0) for block_bsfr in ps_baseline_FR], axis=1).T, \
                   ps_baseline_T
        else:
            return [block_psfr.mean(axis=0, keepdims=True) for block_psfr in ps_FR],\
                   ps_T,\
                   [block_bsfr.mean(axis=0, keepdims=True) for block_bsfr in ps_baseline_FR], \
                   ps_baseline_T


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
        boolArray = lookfor_matching_fname(np.ones(len(matlabfnames), dtype=bool), *uniqueEpoch)
        j = boolArray.array.argmax()
        matlabfnames.iloc[[i, j]] = matlabfnames.iloc[[j, i]]

    pass


if __name__ == '__main__':
    # tms = TMSTG.load(['G:/Vishnu/Analysis/tms-tg/data/SLAnalys.mat'])
    dir_path = r'G:\Vishnu\data\TMSTG\20180922'
    matlabfiles = pd.Series(dir_path + '\\' + f for f in os.listdir(dir_path) if f.endswith('.mat'))
    infofile = [dir_path + '\\' + f for f in os.listdir(dir_path) if f.endswith('.xlsx')]
    tms = TMSTG.load(matlabfiles, infofile[0])
    tms.psfr = {'selectionParams': {'Epoch': {'Region': 'MC'}, 'MT': '>=1'}}
    tms.psfr
    tms.psfr
