import numpy as np
import h5py
from tkinter import filedialog


# TODO: Change to metaclass that assigns h5py datasets as attributes
class MATdata:
    """MATdata

    Enables easy access to data contained in h5py group object

    Parameters
    ----------
    mfile: MATfile object

    Attributes
    ---------
    self.slice can be set to numpy.s_ (slice object) to extract only a
    portion of the data from the hard drive.
    Useful when the data is too large and only a small portion is required.

    Do not instantiate this class directly. Meant to be used by MATfile object.

    """

    def __init__(self, mfile: 'MATfile'):
        # Initialize MATdata object with a MATfile object
        self.matfile = mfile
        self.hdf5 = self.read_hDF5()

        # Set slice attribute to None by default
        self.slice = None

        # List all items in the hdf5 group and clean up non-dataset items
        tmplist = []
        self.hdf5.visit(tmplist.append)         # Recursively access all group members and store their path in tmplist
        self.btree = self.__cleanup(tmplist)    # Store in attribute btree the paths to datasets items only

    def read_hDF5(self):
        # Read the data from the .mat file and return it
        try:
            return self.matfile.f['s']
        except KeyError:
            print(f'.mat-file is not of kind \'{self.matfile.kind}\'')

    def __cleanup(self, tmplist):
        # Filter out non-dataset items from the list
        return [i for i in tmplist if isinstance(self.hdf5[i], h5py.Dataset)]

    def _ipython_key_completions_(self):
        # Enable tab-completion in IPython environments
        return self.btree

    def __getitem__(self, item):
        # Retrieve an item from the hdf5 group
        dset = self.hdf5[item]
        s = dset.shape
        ty = dset.dtype
        out = np.empty(s, dtype=ty)

        if self.slice is None:
            # If no slice is specified, read the entire dataset directly
            dset.read_direct(out)
        else:
            # If a slice is specified, read only the specified portion of the dataset
            dset.read_direct(out, source_sel=self.slice, dest_sel=self.slice)
            # Reset the slice attribute to None after reading
            self.slice = None
        return out

    def __repr__(self):
        # Return a string representation of the MATdata object
        cls = self.__class__.__name__
        return f'{cls}<{self.hdf5}>'

    def __del__(self):
        # delete MATfile object
        del self.matfile


class MATfile:
    """MATfile

    Non-transparent implementation of h5py API for simple access to
    data on the disk without any housekeeping that comes with
    reading a file.

    Parameters
    ----------
    fname: 'hdf5' or 'mat' filename.
        If empty instantiates a GUI for selecting a filename.
    kind: [default: SingleLocation]

    """

    def __init__(self, fname=None, kind="SingleLocation"):
        # Initialize MATfile object with optional parameters: file name and kind of file
        self.kind = kind

        # If a file name is provided, set it as the file name attribute, otherwise prompt user to select a file
        if fname is not None:
            self.fname = fname
        else:
            try:
                self.fname = filedialog.askopenfilename(title="Select a .mat file")
            except ValueError:
                print('File not selected')

    def __repr__(self):
        # Return a string representation of the MATfile object
        cls = self.__class__.__name__
        return f'{cls}<{self.fname}>'

    def __del__(self):
        # Close the file when the MATfile object is deleted
        try:
            self.f.close()
            print(f'{self.fname} closed successfully')
        except AttributeError:
            pass
        finally:
            print(f'{self} is deleted')

    def read(self):
        # Open the .mat file and return a MATdata object containing the data
        self.f = h5py.File(self.fname, 'r')
        return MATdata(self)


if __name__ == '__main__':
    matfile = MATfile()     # MATfile('SLAnalys.mat')

    # reads file using h5py interface
    data = matfile.read()

    del data

    # reads the data from hard disk to memory
    var = data['SpikeModel/Y']

    # navigate through the b-tree to acquire the dataset of interest
    data['SpikeModel/ClusterAssignment/data/'][0, 0]

    # instantiate a h5py object that points to a hdf5/mat file
    hdf5 = h5py.File('SLAnalys.mat', 'r')

    # access the root node
    data = MATdata(hdf5['s'])
