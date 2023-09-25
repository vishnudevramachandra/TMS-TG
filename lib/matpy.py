import numpy as np
import h5py
from tkinter import filedialog

# TODO: Change to metaclass that assigns h5py datasets as attributes
class MATdata(object):
    """Enables easy access to data contained in h5py group object

    Parameters
    ----------
    hdf5: h5py group object containing datasets

    Attributes
    ---------
    self.slice can be set to numpy.s_ (slice object) to extract only a
    portion of the data from the hard drive.
    Useful when the data is too large and only a small portion is required.

    Do not instantiate this class directly. Meant to be used by MATfile object.
    """

    def __init__(self, hdf5):
        self.hdf5 = hdf5
        self.slice = None
        tmplist = []
        hdf5.visit(tmplist.append)
        self.btree = self.__cleanup(tmplist)

    def __cleanup(self, tmplist):
        return [i for i in tmplist if isinstance(self.hdf5[i], h5py.Dataset)]

    def _ipython_key_completions_(self):
        return self.btree

    def __getitem__(self, item):
        dset = self.hdf5[item]
        s = dset.shape
        ty = dset.dtype
        out = np.empty(s, dtype=ty)

        if self.slice == None:
            dset.read_direct(out)
        else:
            dset.read_direct(out, source_sel=self.slice, dest_sel=self.slice)
            self.slice = None
        return out

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}<{self.hdf5}>'


class MATfile(object):
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
        if hasattr(self, "__f"):
            self.__f.close()
        print(f'{self.fname.split("/")[-1]} closed successfully')

    def read(self):
        self.__f = h5py.File(self.fname, 'r')
        return MATdata(self.read_hDF5())

    def read_hDF5(self):
        try:
            return self.__f['s']
        except KeyError:
            self.__f.close()
            raise KeyError(f'.mat-file is not of kind \'{self.kind}\'')
        finally:
            print(f'{self.fname.split("/")[-1]} successfully read')
            # close the file here



if __name__ == '__main__':
    matfile = MATfile()     # MATfile('SLAnalys.mat')

    # reads file using h5py interface
    data = matfile.read()

    # reads the data from hard disk to memory
    var = data['SpikeModel/Y']

    # navigate through the b-tree to acquire the dataset of interest
    data['SpikeModel/ClusterAssignment/data/'][0, 0]

