import numpy as np
import pandas as pd
import lib.matpy as mp
import os
from typing import Optional

layers = {'L23': (200, 800), 'L4': (800, 1200), 'L5': (1200, 1600), 'L6': (1600, 1900)}
regions = {'thal': ['BZ', 'CZ'], 'MC': ['CFA', 'MC'], 'SC': ['S1', 'SC'], 'VC': ['V1', 'VC']}
coilDir = {'ML', 'LM', 'PA', 'AP'}
StimHem = {'LH', 'RH'}
coilPos = {'MC', 'SC', 'VC'}


class TMSTG(object):

    def __init__(self, matdata=None, epochinfo=None) -> None:

        self.matdata: list[mp.MATdata] = matdata if matdata is not None else list()
        self.epochinfo: Optional[pd.DataFrame] = epochinfo

    @classmethod
    def load(cls, matlabfnames: 'pd.Series[str]', infofile: Optional[str] = None) -> 'TMSTG':
        """ create a TMSTG object using a list of singleLocation files and an infofile """

        if infofile is not None:
            epochinfo = pd.read_excel(infofile).dropna()
            epochinfo = cls.do_multi_indexing(epochinfo)
            _sort_filelist(matlabfnames, epochinfo)
        else:
            epochinfo = None

        return cls([mp.MATfile(fname).read() for fname in matlabfnames], epochinfo)

    @staticmethod
    def do_multi_indexing(epochinfo: pd.DataFrame) -> pd.DataFrame:

        df = epochinfo.copy()
        df.loc[:, 'Depth'] = df['Depth'].str.removesuffix('µm')
        df['Depth_int'] = np.int_(df.loc[:, 'Depth'].to_list())

        for key in layers:
            df.loc[(layers[key][0] <= df['Depth_int'])
                   & (df['Depth_int'] < layers[key][1]), 'Layer'] \
                = key

        for key in regions:
            df.loc[df['RecArea '].str.contains('|'.join(regions[key])), 'Region'] = key

        df.loc[df['StimHem'] == df['RecHem'], 'CoilvsRecArea'] = 'ipsi'
        df.loc[df['StimHem'] != df['RecHem'], 'CoilvsRecArea'] = 'contra'
        df.fillna('missing', inplace=True)
        df.set_index(['Region', 'Layer', 'CoilvsRecArea', 'Depth'], inplace=True)
        return df


def _sort_filelist(matlabfnames, epochinfo) -> None:
    """ sorts the order of matlabf<ile>names in the list to be consistent with epoch order """

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
    tms
