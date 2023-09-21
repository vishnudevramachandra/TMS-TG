import numpy as np
import numba as nb
import pandas as pd
import re
from lib.dataanalysis import peristim_firingrate


class PSFR(object):
    """
    set parameters, compute peri-stimulus firing rate, cache it, and append when demanded
    """

    cols_with_floats = {'MSO ', 'MT', 'no. of Trigs', 'Stimpulses', 'Depth_int'}
    cols_with_strings = {'StimHem', 'CoilDir', 'TG-Injection ', 'RecArea ', 'RecHem', 'Filename'}

    def __init__(self):
        self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T \
            = list(), list(), list(), list()

    def __set__(self, obj, psfr_params):
        self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T \
            = list(), list(), list(), list()
        default_params = type(obj).psfr_default_params
        if not hasattr(obj, '_psfr_params'):
            obj._psfr_params = default_params

        try:
            if not isinstance(psfr_params, dict):
                psfr_params = obj._psfr_params
                raise ValueError

            raiseFlag = False
            if 'smoothingParams' in psfr_params.keys():
                unentered_keys = default_params['smoothingParams'].keys() \
                                 - psfr_params['smoothingParams'].keys()
                for key in unentered_keys:
                    psfr_params['smoothingParams'][key] = obj._psfr_params['smoothingParams'][key]
                if unentered_keys == default_params['smoothingParams'].keys():
                    raiseFlag = True
            else:
                psfr_params['smoothingParams'] = obj._psfr_params['smoothingParams']

            if 'selectionParams' in psfr_params.keys():
                if 'Epoch' not in psfr_params['selectionParams'].keys() \
                        or \
                        not issubclass(type(psfr_params['selectionParams']['Epoch']), dict) \
                        or \
                        not (psfr_params['selectionParams']['Epoch'].keys()
                             & default_params['selectionParams']['Epoch'].keys()):
                    psfr_params['selectionParams']['Epoch'] \
                        = obj._psfr_params['selectionParams']['Epoch']
                    raiseFlag = True
                else:
                    for epochKey in default_params['selectionParams']['Epoch']:
                        if epochKey in psfr_params['selectionParams']['Epoch'].keys():
                            if not issubclass(type(psfr_params['selectionParams']['Epoch'][epochKey]),
                                              tuple | set | list | None):
                                psfr_params['selectionParams']['Epoch'][epochKey] \
                                    = (psfr_params['selectionParams']['Epoch'][epochKey],)
                        else:
                            psfr_params['selectionParams']['Epoch'][epochKey] \
                                = obj._psfr_params['selectionParams']['Epoch'][epochKey]
            else:
                psfr_params['selectionParams'] = obj._psfr_params['selectionParams']

            if 'timeWin' in psfr_params.keys():
                if len(psfr_params['timeWin']) == 2:
                    psfr_params['timeWin'] = tuple(psfr_params['timeWin'])
                else:
                    psfr_params['timeWin'] = obj._psfr_params['timeWin']
                    raiseFlag = True
            else:
                psfr_params['timeWin'] = obj._psfr_params['timeWin']

            if 'trigger' in psfr_params.keys():
                # TODO: random trigger implementation
                pass
            else:
                psfr_params['trigger'] = obj._psfr_params['trigger']

            if 'baselinetimeWin' in psfr_params.keys():
                pass
            else:
                psfr_params['baselinetimeWin'] = obj._psfr_params['baselinetimeWin']

            if raiseFlag:
                raise ValueError

        except ValueError:
            print(f'psfr_params does not adhere to correct format, '
                  f'using instead default params...')

        print('psfr_params set to: ', psfr_params)
        obj._psfr_params = psfr_params

    def __get__(self, obj, objtype):
        # obj.matdata[0][obj.matdata[0]['CombiMCD_fnames'].flatten()[0]].tobytes().decode('utf-16')
        # pd.set_option('display.expand_frame_repr', False)

        if len(self._ps_FR) != 0:
            return self._ps_FR, self._ps_T, self._ps_baseline_FR, self._ps_baseline_T

        uniqueEpochs = obj.epochinfo.index.unique().to_frame(index=False)
        selectEpochs = self._filterEpochs(uniqueEpochs, obj._psfr_params['selectionParams']['Epoch'])

        for i, epoch in selectEpochs.items():

            if 'TMS' in obj._psfr_params['trigger'].keys():
                trigger = self._read_trigger(obj.matdata[i])
            else:
                ...
            # TODO: random trigger implementation

            singleUnitsSpikeTimes = self._read_singleUnitsSpikeTimes(obj.matdata[i])

            thisEpoch_df = obj.epochinfo.loc[epoch, :]
            num_of_trigs = thisEpoch_df['no. of Trigs'].to_numpy(dtype=int)
            assert sum(num_of_trigs) == len(trigger)  # that no. of triggers in infofile matches with the data
            mso = self._read_MSO(obj.matdata[i])
            assert all(thisEpoch_df['MSO '] == mso)  # that the mso order in infofile matches with the data

            # initialize boolArray[True] for selecting (booleanIndexing) blocks using criterion in ['selectionParams']
            idx = thisEpoch_df['MSO '] == thisEpoch_df['MSO ']

            # change the truth values of Index by doing floating point comparison
            selectCols = obj._psfr_params['selectionParams'].keys() & self.cols_with_floats
            for col in selectCols:
                string = obj._psfr_params['selectionParams'][col]
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
            selectCols = obj._psfr_params['selectionParams'].keys() & self.cols_with_strings
            for col in selectCols:
                string = obj._psfr_params['selectionParams'][col]
                idx &= thisEpoch_df[col].str.contains(string)

            # select trigger by using the Index
            startTrigIdx, stopTrigIdx = np.append(0, num_of_trigs[:-1].cumsum()), num_of_trigs.cumsum()
            startTrigIdx, stopTrigIdx = startTrigIdx[idx], stopTrigIdx[idx]
            trigIdx_blockwise = [np.arange(x, y).tolist() for x, y in zip(startTrigIdx, stopTrigIdx)]
            selectTrigger = trigger[[item for row in trigIdx_blockwise for item in row]]

            # using selected trigger compute peristimulus FiringRate
            if selectTrigger.size != 0:
                timeIntervals = self._compute_timeIntervals(selectTrigger, *obj._psfr_params['timeWin'])

                tmp_ps_FR, self._ps_T = peristim_firingrate(
                    singleUnitsSpikeTimes, timeIntervals, obj._psfr_params['smoothingParams'])
                self._ps_FR.append(tmp_ps_FR)

                timeIntervals_baseline, baselineWinWidth \
                    = self._compute_timeIntervals_baseline(selectTrigger, *obj._psfr_params['baselinetimeWin'])

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
