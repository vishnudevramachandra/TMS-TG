"""
constants.py
    contains the necessary constants for tms_tg.py
"""

LAYERS = {'MC': {'L23': (200, 650), 'L4': (650, 800), 'L5': (800, 1600), 'L6': (1600, 2000)},
          'SC': {'L23': (200, 600), 'L4': (600, 900), 'L5': (900, 1600), 'L6': (1600, 2000)},
          'VC': {'L23': (200, 600), 'L4': (600, 900), 'L5': (900, 1600), 'L6': (1600, 2000)}}
REGIONS = {'thal': ['BZ', 'CZ', 'CL', 'PC', 'VM', 'VPM', 'VPL', 'ZI', 'PO', 'SubV', 'MDL'],
           'BG': ['STN'],
           'MC': ['CFA', 'MC', 'M1'],
           'SC': ['S1', 'SC'],
           'VC': ['V1', 'VC']}
COILDIR = {'ML', 'LM', 'PA', 'AP'}
STIMHEM = {'LH', 'RH'}
COILPOS = {'MC', 'SC', 'VC'}
MANIPULATION = {'MUS'}

EPOCHISOLATORS = ['Animal', 'Region', 'Layer', 'CoilHemVsRecHem', 'Mov', 'Depth']
COLS_WITH_NUMS = {'MSO ', 'MT', 'no. of Trigs', 'Stimpulses', 'Depth_int'}
COLS_WITH_PURE_STRINGS = {'StimHem', 'CoilDir', 'RecArea ', 'RecHem', 'CHvsRH', 'TGcut'}
COLS_WITH_STRINGS_AND_DIGITS = {'TG-Injection ', 'Filename', 'Skin-Injection', 'Skin-Injection',
                                'TGOrifice', 'CortexAbl'}
