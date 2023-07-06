import numpy as np
import h5py

class MATdata:
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



if __name__ == '__main__':

    # instantiate a h5py object that points to a hdf5/mat file
    hdf5 = h5py.File('SLAnalys.mat', 'r')

    data = MATdata(hdf5['s'])

    # navigate through the b-tree to acquire the dataset of interest
    data['SpikeModel/Y']
