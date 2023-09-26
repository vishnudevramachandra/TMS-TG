import numpy as np
import numba as nb
import re
from lib.dataanalysis import peristim_firingrate
from collections.abc import Iterator


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
                                                  tuple | set | list | None):
                                    params['selectionParams']['Epoch'][epochKey] \
                                        = (params['selectionParams']['Epoch'][epochKey],)
                            else:
                                params['selectionParams']['Epoch'][epochKey] \
                                    = self.analysis_params['selectionParams']['Epoch'][epochKey]
                else:
                    params['selectionParams'] = self.analysis_params['selectionParams']

                if 'smoothingParams' in params.keys():
                    unentered_keys = self.analysis_params['smoothingParams'].keys() \
                                     - params['smoothingParams'].keys()
                    for key in unentered_keys:
                        params['smoothingParams'][key] = self.analysis_params['smoothingParams'][key]
                    if unentered_keys == self.analysis_params['smoothingParams'].keys():
                        raiseFlag = True
                else:
                    params['smoothingParams'] = self.analysis_params['smoothingParams']

                if 'timeWin' in params.keys():
                    if len(params['timeWin']) == 2:
                        params['timeWin'] = tuple(params['timeWin'])
                    else:
                        params['timeWin'] = self.analysis_params['timeWin']
                        raiseFlag = True
                else:
                    params['timeWin'] = self.analysis_params['timeWin']

                if 'trigger' in params.keys():
                    # TODO: random trigger implementation
                    pass
                else:
                    params['trigger'] = self.analysis_params['trigger']

                if 'baselinetimeWin' in params.keys():
                    pass
                else:
                    params['baselinetimeWin'] = self.analysis_params['baselinetimeWin']

                if raiseFlag:
                    raise ValueError

            except ValueError:
                print(f'psfr_params does not adhere to correct format, '
                      f'using instead default/previous params...')

            print('psfr_params set to: ', params)
            self.analysis_params = params
            obj.psfr = list(), list(), list(), list()
            del obj._filter_blocks
            obj._filter_blocks

    def __get__(self, obj, objType):
        return self.analysis_params


class PSFR(object):
    """
    Compute peri-stimulus firing rate, cache it, and append when demanded
    """

    def __init__(self):
        self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T \
            = list(), list(), list(), list()

    def __set__(self, obj, *args):
        self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T \
            = args[0][0], args[0][1], args[0][2], args[0][3]

    def __get__(self, obj, objType):
        # obj.matdata[0][obj.matdata[0]['CombiMCD_fnames'].flatten()[0]].tobytes().decode('utf-16')
        # pd.set_option('display.expand_frame_repr', False)

        if len(self._ps_FR) != 0:
            return self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T

        if 'TrigStartIdx' not in obj.epochinfo.columns:
            _concatEpochinfo(obj.matdata, obj.epochinfo, 'TrigStartIdx')
            _check_mso_order(obj.matdata, obj.epochinfo)

        selectBlocks, blockIdx = obj._filter_blocks

        for rowIndex, row in selectBlocks.iterrows():
            if 'TMS' in obj.analysis_params['trigger'].keys():
                trigger = _read_trigger(obj.matdata[rowIndex])
            else:
                ...
            # TODO: random trigger implementation

            # select trigger using 'TrigStartIdx' in epochinfo
            selectTrigger = trigger[row['TrigStartIdx']: row['TrigStartIdx'] + row['no. of Trigs']]

            timeIntervals = _compute_timeIntervals(selectTrigger, *obj.analysis_params['timeWin'])
            tmp_ps_FR, self._ps_T = peristim_firingrate(
                obj.singleUnitsSpikeTimes[rowIndex], timeIntervals, obj.analysis_params['smoothingParams'])
            self._ps_FR.append(tmp_ps_FR)

            timeIntervals_baseline, baselineWinWidth \
                = _compute_timeIntervals_baseline(selectTrigger, *obj.analysis_params['baselinetimeWin'])

            tmp_ps_FR, self._ps_baseline_T = peristim_firingrate(
                obj.singleUnitsSpikeTimes[rowIndex], timeIntervals_baseline,
                {'win': 'rect', 'width': baselineWinWidth, 'overlap': 0.0})
            self._ps_baseline_FR.append(tmp_ps_FR)

        return self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T


def _compute_timeIntervals(trigger, startT, endT):
    timeIntervals = nb.typed.List()
    [timeIntervals.append((x + startT, x + endT)) for x in trigger]
    return timeIntervals


def _compute_timeIntervals_baseline(trigger, startT, endT):
    timeIntervals = nb.typed.List()
    baselineWinWidth = endT - startT
    [timeIntervals.append((x + startT + (baselineWinWidth / 2), x + endT)) for x in trigger]
    return timeIntervals, baselineWinWidth


def _concatEpochinfo(matdata, epochinfo, colName):
    if colName == 'TrigStartIdx':
        epochinfo['TrigStartIdx'] = 0
        uniqueEpochs = epochinfo.index.unique().to_numpy()
        for uniqueEpoch in uniqueEpochs:
            trigger = _read_trigger(matdata[uniqueEpoch])
            num_of_trigs = epochinfo.loc[uniqueEpoch, 'no. of Trigs'].to_numpy(dtype=int)
            assert sum(num_of_trigs) == len(trigger), 'no. of triggers in infofile does not match with mat-data'
            epochinfo.loc[uniqueEpoch, 'TrigStartIdx'] = np.append(0, num_of_trigs.cumsum()[:-1])


def _read_trigger(data):
    trigChanIdx = data['TrigChan_ind'][0, 0].astype(int) - 1
    refs = data['rawData/trigger'].flatten()
    return [data[i].flatten() for i in refs][trigChanIdx][::2] * 1e3


def _check_mso_order(matdata, epochinfo):
    uniqueEpochs = epochinfo.index.unique().to_numpy()
    for uniqueEpoch in uniqueEpochs:
        mso = _read_MSO(matdata[uniqueEpoch])
        assert all(epochinfo.loc[uniqueEpoch, 'MSO '] == mso), 'mso order in infofile differs from mat-data'


def _read_MSO(data):
    refs = data['blockInfo/MSO'].flatten()
    mso = [data[i].flatten().tobytes().decode('utf-16') for i in refs]
    return [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else 0 for item in mso]


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
