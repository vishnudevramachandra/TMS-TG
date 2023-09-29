import numpy as np
import pandas as pd
import re
from lib.dataanalysis import peristim_firingrate, peristim_timestamp
from collections.abc import Iterator
from functools import lru_cache


class AnalysisParams(object):
    """
    Set parameters for selecting a subset of data and analysing it
    """

    def __init__(self, params):
        self.__set__(self, params)

    def __set__(self, obj, params):
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

                if 'lateComponentParams' in params.keys():
                    unentered_keys = self.analysis_params['lateComponentParams'].keys() \
                                     - params['lateComponentParams'].keys()
                    for key in unentered_keys:
                        params['lateComponentParams'][key] = self.analysis_params['lateComponentParams'][key]
                    if unentered_keys == self.analysis_params['lateComponentParams'].keys():
                        raiseFlag = True
                else:
                    params['lateComponentParams'] = self.analysis_params['lateComponentParams']

                if raiseFlag:
                    raise ValueError

            except ValueError:
                print(f'psfr_params does not adhere to correct format, '
                      f'using instead default/previous params...')

            print('psfr_params set to: ', params)
            self.analysis_params = params

            # do housekeeping
            obj.psfr = list(), list(), list(), list()
            obj.Raster = list()
            obj.filter_blocks = None
            _, _ = obj.filter_blocks

    def __get__(self, obj, objType):
        return self.analysis_params


class PSFR(object):
    """
    Compute peri-stimulus firing rate and cache it when demanded
    """

    def __init__(self):
        self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T \
            = list(), list(), list(), list()

    def __set__(self, obj, *args):
        self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T \
            = args[0][0], args[0][1], args[0][2], args[0][3]

    def __get__(self, obj, objType) -> tuple[list, list, list, list]:

        if len(self._ps_FR) != 0:
            return self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T

        _check_trigger_numbers(obj.matdata, obj.blocksinfo)
        _check_mso_order(obj.matdata, obj.blocksinfo)
        selectBlocks, selectBlocksIdx = obj.filter_blocks

        for epochIndex, blockinfo in selectBlocks.iterrows():
            selectTrigger = _get_trigger_times_for_current_block(
                obj.analysis_params['peristimParams']['trigger'], obj.matdata[epochIndex], blockinfo)

            timeIntervals = _compute_timeIntervals(
                selectTrigger, *obj.analysis_params['peristimParams']['timeWin'])
            tmp_ps_FR, self._ps_T \
                = peristim_firingrate(obj.singleUnitsSpikeTimes(epochIndex),
                                      timeIntervals,
                                      obj.analysis_params['peristimParams']['smoothingParams'])
            self._ps_FR.append(tmp_ps_FR)

            timeIntervals_baseline, baselineWinWidth = _compute_timeIntervals_baseline(
                selectTrigger, *obj.analysis_params['peristimParams']['baselinetimeWin'])
            tmp_ps_FR, self._ps_baseline_T \
                = peristim_firingrate(obj.singleUnitsSpikeTimes(epochIndex),
                                      timeIntervals_baseline,
                                      {'win': 'rect', 'width': baselineWinWidth, 'overlap': 0.0})
            self._ps_baseline_FR.append(tmp_ps_FR)

        return self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T


class Raster(object):
    """
    Compute peri-stimulus spike time-stamps and cache it when demanded
    """

    def __init__(self):
        self._ps_TS = list()

    def __set__(self, obj, value):
        self._ps_TS = value

    def __get__(self, obj, objType) -> list:

        if len(self._ps_TS) != 0:
            return self._ps_TS

        _check_trigger_numbers(obj.matdata, obj.blocksinfo)
        _check_mso_order(obj.matdata, obj.blocksinfo)
        selectBlocks, selectBlocksIdx = obj.filter_blocks

        for epochIndex, blockinfo in selectBlocks.iterrows():
            selectTrigger = _get_trigger_times_for_current_block(
                obj.analysis_params['peristimParams']['trigger'], obj.matdata[epochIndex], blockinfo)
            timeIntervals = _compute_timeIntervals(
                selectTrigger, *obj.analysis_params['peristimParams']['timeWin'])
            tmp_ps_TS = peristim_timestamp(obj.singleUnitsSpikeTimes(epochIndex), timeIntervals)
            self._ps_TS.append(tmp_ps_TS)

        return self._ps_TS


def _get_trigger_times_for_current_block(trigParams, matdatum, blockinfo):
    if 'TMS' in trigParams.keys():
        trigger = _read_trigger(matdatum)
        return trigger[blockinfo['TrigStartIdx'] + np.array(range(blockinfo['no. of Trigs']))]
    else:
        ...  # TODO: random trigger implementation


def _compute_timeIntervals(trigger, startT, endT):
    return trigger[:, np.newaxis] + np.array([startT, endT])


def _compute_timeIntervals_baseline(trigger, startT, endT):
    baselineWinWidth = endT - startT
    return trigger[:, np.newaxis] + np.array([startT + (baselineWinWidth / 2), endT]), baselineWinWidth


@lru_cache(maxsize=None)
def _read_trigger(matdatum):
    trigChanIdx = matdatum['TrigChan_ind'][0, 0].astype(int) - 1
    refs = matdatum['rawData/trigger'].flatten()
    return matdatum[refs[trigChanIdx]].flatten()[::2] * 1e3


def _check_trigger_numbers(matdata, blocksinfo):
    epochIndices = blocksinfo.index.unique().to_numpy()
    for epochIndex in epochIndices:
        trigger = _read_trigger(matdata[epochIndex])
        assert blocksinfo.loc[epochIndex, 'no. of Trigs'].sum() == len(trigger), \
            f'no. of triggers in epoch {epochIndex} does not match with mat-data'
        # matdata[epochIndex][matdata[epochIndex]['CombiMCD_fnames'].flatten()[0]].tobytes().decode('utf-16')


@lru_cache(maxsize=None)
def _read_MSO(matdatum):
    refs = matdatum['blockInfo/MSO'].flatten()
    mso = [matdatum[i].flatten().tobytes().decode('utf-16') for i in refs]
    return np.array(
        [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else 0 for item in mso])


def _check_mso_order(matdata, blocksinfo):
    epochIndices = blocksinfo.index.unique().to_numpy()
    for epochIndex in epochIndices:
        mso = _read_MSO(matdata[epochIndex])
        nonZeroMSOindices = blocksinfo.loc[epochIndex, 'MSO '].to_numpy() != 0
        assert all(blocksinfo.loc[epochIndex, 'MSO '][nonZeroMSOindices]
                   == mso[nonZeroMSOindices]), \
            f'mso order in epoch {epochIndex} differs from mat-data'


class LateComponent(object):

    def __init__(self):
        self.attr = 5

    def __set__(self, obj, value):
        self.attr = value

    def __call__(self, *args, **kwargs):
        print(*args)

    def method(self):
        ...


class SpikeTimes(Iterator):

    def __call__(self, *args, **kwargs):
        ...

    def __get__(self, obj, objType):
        ...

    def __init__(self):
        self._index = 0

    def __next__(self):
        if self._index < len(self._index):
            self._index += 1
            return 'item'
        else:
            self._index = 0
            raise StopIteration

    def __set__(self, obj, value):
        ...
