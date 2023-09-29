import matplotlib.pyplot as plt
import numpy as np
import re
from tms_tg import TMSTG, EPOCHISOLATORS, COLS_WITH_FLOATS
from itertools import zip_longest
from cachetools import cached
from cachetools.keys import hashkey
from functools import lru_cache

def raster_plot(**kwargs):
    rows = {'raster', 'psfr'}
    keys = kwargs.keys()

    for key in kwargs:
        if key == 'raster':
            1


def ascertain_analysis_params_from_colNames(colNames):
    analysis_params = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ]))}}
    if colNames is None:
        colNames = [{'All': None}]
    else:
        keys = list(colNames[0].keys())
        colNamesAsSingleDictItem = {key: [] for key in keys}
        for key in set(keys) & set(EPOCHISOLATORS):
            for i in range(len(colNames)):
                if type(colNames[i][key]) != dict:
                    colNamesAsSingleDictItem[key].append(colNames[i][key])
                else:
                    key, value = list(colNames[i][key].items())[0]
                    if key == 'subRegion' and re.search('Thal', value):
                        colNamesAsSingleDictItem[key].append('thal')
                    else:
                        raise NotImplementedError(f'dict item {colNames[i][key]} for: {key}')

            analysis_params['selectionParams']['Epoch'][key] = np.unique(colNamesAsSingleDictItem[key])
    return colNames, analysis_params


@cached(cache={}, key=lambda tms, analysis_params, uniqueEpochs: hashkey(uniqueEpochs))
def compute_raster(tms, analysis_params, uniqueEpochs):
    tms.analysis_params = analysis_params
    return tms.psts

def matching_blocks(selectBlocks, colName, cond):
    ascertain_analysis_params_from_colNames(colName)
    index = selectBlocks.index
    return selectBlocks, index


def plot(tms, colNames=None, raster=False, psfr=False, global_avg_FR=False):

    assert type(colNames) == tuple or colNames is None, 'colNames has to be a list of dictionary items or None'

    colNames, analysis_params = ascertain_analysis_params_from_colNames(colNames)

    if raster is False and psfr is False:
        pass
    elif raster is True and psfr is False:
        ps_TS = compute_raster(tms, analysis_params, tuple(selectBlocks.index.unique().to_numpy()))
        selectBlocks, blockIdx = tms.filter_blocks
        fig, ax = plt.subplots(3, len(colNames))
        rowCond = ({'MT': '<1'}, {'MT': '==1'}, {'MT' '>1'})
        for i in range(3):
            for j in range(len(colNames)):
                _, idx = matching_blocks(selectBlocks, colNames[j], rowCond[i])
                1

    elif raster is False and psfr is True:
        ...
    elif raster is True and psfr is True:
        ps_TS = compute_raster(tms, analysis_params)
        selectBlocks, blockIdx = tms.filter_blocks


if __name__ == '__main__':
    cols = ({'Region': 'MC', 'Layer': 'L5'},
            {'Region': 'SC', 'Layer': 'L5'},
            {'Region': {'subRegion': 'MotorThal'}, 'Layer': 'none'},
            {'Region': {'subRegion': 'SensoryThal'}, 'Layer': 'none'})
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    plot(tms, colNames=cols, raster=True, psfr=False, global_avg_FR=False)