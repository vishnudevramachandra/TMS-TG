import numpy as np
import numba as nb
import pandas as pd
import re
from lib.dataanalysis import peristim_firingrate
from collections.abc import Iterator

COLS_WITH_FLOATS = {'MSO ', 'MT', 'no. of Trigs', 'Stimpulses', 'Depth_int'}
COLS_WITH_STRINGS = {'StimHem', 'CoilDir', 'TG-Injection ', 'RecArea ', 'RecHem', 'Filename'}


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

        uniqueEpochs = obj.epochinfo.index.unique().to_frame(index=False)
        selectEpochs = self._filterEpochs(uniqueEpochs, obj.analysis_params['selectionParams']['Epoch'])

        for i, epoch in selectEpochs.items():

            if 'TMS' in obj.analysis_params['trigger'].keys():
                trigger = self._read_trigger(obj.matdata[i])
            else:
                ...
            # TODO: random trigger implementation

            singleUnitsSpikeTimes = self._read_singleUnitsSpikeTimes(obj.matdata[i])

            thisEpoch_df = obj.epochinfo.loc[epoch, :]
            num_of_trigs = thisEpoch_df['no. of Trigs'].to_numpy(dtype=int)
            assert sum(num_of_trigs) == len(trigger), 'no. of triggers in infofile does not match with mat-data'
            mso = self._read_MSO(obj.matdata[i])
            assert all(thisEpoch_df['MSO '] == mso), 'mso order in infofile differs from mat-data'

            # initialize boolArray[True] for selecting (booleanIndexing) blocks using criterion in ['selectionParams']
            idx = thisEpoch_df['MSO '] == thisEpoch_df['MSO ']

            # change the truth values of Index by doing floating point comparison
            selectCols = obj.analysis_params['selectionParams'].keys() & COLS_WITH_FLOATS
            for col in selectCols:
                string = obj.analysis_params['selectionParams'][col]
                if re.match('<=', string):
                    val = re.sub('<=', '', string)
                    idx &= thisEpoch_df[col] <= np.float_(val)
                elif re.match('<', string):
                    val = re.sub('<', '', string)
                    idx &= thisEpoch_df[col] <= np.float_(val)
                elif re.match('>=', string):
                    val = re.sub('>=', '', string)
                    idx &= thisEpoch_df[col] >= np.float_(val)
                elif re.match('>', string):
                    val = re.sub('>', '', string)
                    idx &= thisEpoch_df[col] > np.float_(val)
                elif re.match('==', string):
                    val = re.sub('==', '', string)
                    idx &= thisEpoch_df[col] == np.float_(val)

            # change the truth values of Index by doing string comparison
            selectCols = obj.analysis_params['selectionParams'].keys() & COLS_WITH_STRINGS
            for col in selectCols:
                string = obj.analysis_params['selectionParams'][col]
                idx &= thisEpoch_df[col].str.contains(string)

            # select trigger by using the Index
            startTrigIdx, stopTrigIdx = np.append(0, num_of_trigs[:-1].cumsum()), num_of_trigs.cumsum()
            startTrigIdx, stopTrigIdx = startTrigIdx[idx], stopTrigIdx[idx]
            trigIdx_blockwise = [np.arange(x, y).tolist() for x, y in zip(startTrigIdx, stopTrigIdx)]
            selectTrigger = trigger[[item for row in trigIdx_blockwise for item in row]]

            # using selected trigger compute peristimulus FiringRate
            if selectTrigger.size != 0:
                timeIntervals = self._compute_timeIntervals(selectTrigger, *obj.analysis_params['timeWin'])

                tmp_ps_FR, self._ps_T = peristim_firingrate(
                    singleUnitsSpikeTimes, timeIntervals, obj.analysis_params['smoothingParams'])
                self._ps_FR.append(tmp_ps_FR)

                timeIntervals_baseline, baselineWinWidth \
                    = self._compute_timeIntervals_baseline(selectTrigger, *obj.analysis_params['baselinetimeWin'])

                tmp_ps_FR, self._ps_baseline_T = peristim_firingrate(
                    singleUnitsSpikeTimes, timeIntervals_baseline,
                    {'win': 'rect', 'width': baselineWinWidth, 'overlap': 0.0})
                self._ps_baseline_FR.append(tmp_ps_FR)

        return self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T

    @staticmethod
    def _filterEpochs(uniqueEpochs, epochSelectors):
        idx = pd.Series(np.ones(uniqueEpochs.shape[0], dtype=bool))
        for key in epochSelectors:
            if epochSelectors[key] is not None:
                idx &= uniqueEpochs[key].str.contains('|'.join(epochSelectors[key]))
        selectEpochs = uniqueEpochs.loc[idx, :]
        return pd.Series([tuple(x) for x in selectEpochs.to_numpy()], index=selectEpochs.index)

    @staticmethod
    def _read_MSO(data):
        refs = data['blockInfo/MSO'].flatten()
        mso = [data[i].flatten().tobytes().decode('utf-16') for i in refs]
        return [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else 0 for item in mso]

    @staticmethod
    def _read_trigger(data):
        trigChanIdx = data['TrigChan_ind'][0, 0].astype(int) - 1
        refs = data['rawData/trigger'].flatten()
        return [data[i].flatten() for i in refs][trigChanIdx][::2] * 1e3

    @staticmethod
    def _compute_timeIntervals(trigger, startT, endT):
        timeIntervals = nb.typed.List()
        [timeIntervals.append((x + startT, x + endT)) for x in trigger]
        return timeIntervals

    @staticmethod
    def _compute_timeIntervals_baseline(trigger, startT, endT):
        timeIntervals = nb.typed.List()
        baselineWinWidth = endT - startT
        [timeIntervals.append((x + startT + (baselineWinWidth / 2), x + endT)) for x in trigger]
        return timeIntervals, baselineWinWidth

    @staticmethod
    def _read_singleUnitsSpikeTimes(matdata):
        multiUnitSpikeTimes: np.ndarray = matdata['SpikeModel/SpikeTimes/data'].flatten()
        refs = matdata['SpikeModel/ClusterAssignment/data'].flatten()
        allSpikeTimes_neuronIdx = [matdata[i].flatten().astype(int) - 1 for i in refs]
        singleUnitsSpikeTimes = nb.typed.List()
        [singleUnitsSpikeTimes.append(multiUnitSpikeTimes[idx]) for idx in allSpikeTimes_neuronIdx]
        return singleUnitsSpikeTimes


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
