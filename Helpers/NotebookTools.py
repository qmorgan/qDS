import numpy as np
import sys
import time
import os


class ProgressChecker:
    '''Print the progress of a for-loop to stdout

    Arguments:
      iterable (list): the object being iterated over (e.g., a list)
      n_checkpoints (int, optional): the number of update points to print
        out. For instance, n_checkpoints=500 will update every ~0.2%

    Methods:
      check()
      reset()

    Attributes:
      count
      checkcount
      total
      n_checkpoints
      last_updated_time
      timedelta
      elapsedtime
      etd
      checkpoints

    Example Usage:
        import time
        numlist = range(1522)
        progress = ProgressChecker(numlist)
        for num in numlist:
            time.sleep(0.002)
            progress.check(eta=True)
    '''
    def __init__(self, iterable, n_checkpoints=500):
        self.count = 0
        self.checkcount = 0
        self.total = int(len(iterable))
        self.n_checkpoints = int(n_checkpoints)
        self.last_updated_time = None
        self.timedelta = 0
        self.elapsedtime = 0
        self.etd = 0
        # generate evenly space checkpoints based on desired number
        checkpoints = np.linspace(0, self.total, self.n_checkpoints)
        # convert checkpoints into integers, make into set (hashtable)
        self.checkpoints = set(np.floor(checkpoints).astype(int))

    def check(self, eta=False):
        '''Determine the completeness of the loop.  If eta=True, also
        print out the estimated time till completeness.
        '''
        self.count += 1
        # if np.mod(self.count, self.update) == 0:
        if self.count in self.checkpoints:
            self.checkcount += 1
            frac_done = float(self.count)/self.total
            # TODO: Add time remaining estimation.
            if eta is True:
                cur_time = time.time()
                if self.last_updated_time is not None:
                    self.timedelta = cur_time - self.last_updated_time
                    self.elapsedtime += self.timedelta
                    self.etd = (self.n_checkpoints -
                                self.checkcount)*self.timedelta
                self.last_updated_time = cur_time
                wstr = "\r{0:.1f}% | Runtime:{1:.1f}s | ETD:{2:.1f}s".format(
                    100*frac_done, self.elapsedtime, self.etd)
            else:
                wstr = "\r{0:.1f} %".format(100*frac_done)
            sys.stdout.write(wstr)
            sys.stdout.flush()

    def reset(self):
        '''Set the count back to zero.'''
        self.count = 0


class Paths:
    '''Class to initialize and store filepath locations.

    Arguments:
      usecase_name (str): The subfolder for each usecase with the
        following folder architecture:
          $ALBEADO_DATA/<usecase_name>/
          $ALBEADO_DATA/<usecase_name>/raw
          $ALBEADO_DATA/<usecase_name>/prereduced
          $ALBEADO_DATA/<usecase_name>/reduced

    Methods:
      add_filepath()
      add_subpath()
      check()

    Attributes:
      usecase_name
      base
      raw
      prereduced
      reduced
      pathdict

    Example Usage:
      >>> # Initialize paths
      >>> path = Paths('Drier')
      PATH /workspace/localdata//Drier/raw/ DOES NOT EXIST.
        * Trying to create it..
        * Successfully created empty directory raw:
          /workspace/localdata//Drier/raw/
      base: /workspace/localdata//Drier (successfully initialized)
      reduced: /workspace/localdata//Drier/reduced/ (successfully initialized)
      prereduced: /workspace/localdata//Drier/prereduced/
        (successfully initialized)

      >>> # Print out the prereduced path
      >>> path.prereduced
      '/workspace/localdata//Drier/prereduced/'

      >>> # Add a new path
      >>> path.add_filepath('myfile',
                             'filename.pkl',
                             base='reduced')
      path: /workspace/localdata//Drier/reduced//filename.pkl
        (successfully initialized)
      >>> path.myfile
      '/workspace/localdata//Drier/reduced//filename.pkl'

      >>> path.pathdict
      {'base': '/workspace/localdata//Drier',
       'myfile': '/workspace/localdata//Drier/reduced//filename.pkl',
       'prereduced': '/workspace/localdata//Drier/prereduced/',
       'raw': '/workspace/localdata//Drier/raw/',
       'reduced': '/workspace/localdata//Drier/reduced/'}

    '''
    def __init__(self, usecase_name):
        # Check if the env var is set that says where the data is stored
        if "ALBEADO_DATA" not in os.environ:
            print "You need to set the environment variable ALBEADO_DATA"
            print "to point to the path where your albeado data is stored"
            raise Exception
        # Paths
        self.usecase_name = usecase_name.strip()
        self.pathdict = {}
        self.base = os.environ.get("ALBEADO_DATA") + '/' + self.usecase_name
        self.pathdict.update({'base': self.base})
        for name in ['raw', 'reduced', 'prereduced']:
            self.add_subpath(name, check=False)
        self.check()

    def add_subpath(self, name, base='base', check=True, create=False,
                    overwrite=False):
        '''Add another subdirectory beyond the default ones created when the 
        Paths object was initialized.  
        
        Default subpaths created are 
          * <usecase_name>/raw/
          * <usecase_name>/reduced/
          * <usecase_name>/prereduced/
        
        Args:
          name (str): Desired name of subpath 
          base (str, optional): Name of parent path. Must be either 'base' or
            a subpath that already has been added 
          check (bool, optional): If True, run self.check() on new path
          create (bool, optional): Keyword for self.check()
          overwrite (bool, optional): If subpath attribute already exists for
            the Paths object, overwrite it. 
        
        '''
        if not hasattr(self, name):
            # TODO: Check if base exists before attempting to get it.
            pathstr = "{0}/{1}/".format(getattr(self, base), name.strip())
            setattr(self, name, pathstr)
            self.pathdict.update({name: pathstr})
        elif overwrite is True:
            print "WARNING: {} already defined".format(name)
            print " but overwrite == True; overwriting"
            setattr(self, name, pathstr)
            self.pathdict.update({name: pathstr})
        else:
            print "WARNING: {} already defined; not overwriting".format(name)
            pathstr = getattr(self, name)
        if check:
            self.check(checklist=[pathstr], create=create)

    def add_filepath(self, name, filename, base='base', check=True):
        '''Add a filename to the Paths object.
        
        Args:
          name (str): Desired name of the path attribute
          base (str, optional): Name of parent path. Must be either 'base' or
            a subpath that already has been added 
          check (bool, optional): If True, run self.check() on new path
        
        Example usage
          >>> # Add a new path
          >>> path.add_filepath('myfile',
                                'filename.pkl',
                                 base='reduced')
          path: /workspace/localdata//Drier/reduced//filename.pkl
            (successfully initialized)
          >>> path.myfile
          '/workspace/localdata//Drier/reduced//filename.pkl'
        '''
        
        if not hasattr(self, name):
            # TODO: check if base exists 
            pathstr = "{0}/{1}".format(getattr(self, base), filename.strip())
            setattr(self, name, pathstr)
            self.pathdict.update({name: pathstr})
        else:
            print "WARNING: {} already defined. Not overwriting".format(name)
            pathstr = getattr(self, name)
        if check:
            self.check(checklist=[pathstr], create=False)

    def check(self, checklist=None, create=True):
        '''Check if each path in checklist already exists, and optionally 
        create it. 
        
        Args:
          checklist (list, optional): list of full paths to check. If None,
            check every item currently in the pathdict.
          create (bool, optional): If True, attempt to create an empty 
            directory for all items in the checklist that do not already
            exist.
        '''
        if checklist is not None:
            iters = zip(['path']*len(checklist), checklist)
        else:
            iters = self.pathdict.iteritems()

        for name, path in iters:
            if not os.path.exists(path):
                print "PATH {} DOES NOT EXIST.".format(path)

                if create:
                    print "  * Trying to create it.."
                    try:
                        os.mkdir(path)
                        print "  * Successfully created empty directory:"
                        print "    {}: {}".format(name, path)
                    except:
                        print "  * ERROR: Cannot create {}".format(path)
                else:
                    print " create = False; Not trying to create it. "
            else:
                print "{}: {} (successfully initialized)\n".format(name, path)


def print_memory():
    """ If on linux machine, get node total memory and memory usage.

    Arguments:
      None

    Raises:
      IOError: If cannot open('/proc/meminfo', 'r')

    Example Usage:
      >>> print_memory()
      total: 15.666 GB
      used: 0.363 GB
      free: 15.304 GB
    """
    try:
        with open('/proc/meminfo', 'r') as mem:
            ret = {}
            tmp = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) == 'MemTotal:':
                    ret['total'] = int(sline[1])
                elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    tmp += int(sline[1])
            ret['free'] = tmp
            ret['used'] = int(ret['total']) - int(ret['free'])

        for key, val in ret.iteritems():
            print "{0}: {1:.3f} GB".format(key, val/1e6)

    except IOError:
        print "Cannot access /proc/meminfo. Are you on linux?"
