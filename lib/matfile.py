import numpy as np
import pandas as pd
import h5py
from tkinter import filedialog
from matdata import MATdata

class MATfile:
    """MATfile

    Non-transparent implementation of h5py API for simple access to
    data on the disk without any housekeeping that comes with
    reading a file.

    Parameters
    ----------
    fname: 'hdf5' or 'mat' filename.
        If empty instantiates a GUI for selecting a filename.

    """
    def __init__(self, fname=None, kind="SingleLocation"):
        self.kind = kind

        if fname != None:
            self.fname = fname
        else:
            try:
                self.fname = filedialog.askopenfilename(title="Select a .mat file")
            except ValueError:
                print('File not selected')

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}<{self.fname}>'

    def __del__(self):
        self.__f.close()
        print(f'{self.fname} closed successfully')

    def read(self):
        self.__f = h5py.File(self.fname, 'r')
        return MATdata(self.read_hDF5())

    def read_hDF5(self):
        try:
            return self.__f['s']
        except KeyError:
            print(f'.mat-file is not of kind \'{self.kind}\'')
        finally:
            print('Matlab file successfully read')
            # close the file here



if __name__ == '__main__':
    matfile = MATfile()

    # reads file using h5py interface
    data = matfile.read()

    # reads the data from hard disk to memory
    var = data['SpikeModel/Y']

