import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import os

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import (gb_addinfo, gp_extractor, pkFr_extr, delay_extr, compute_excludeIdx, violin_fill_false)

# TODO: plot the change of late activity component after TG-silencing and PCA plot