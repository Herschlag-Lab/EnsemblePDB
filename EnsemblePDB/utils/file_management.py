""" EnsemblePDB.utils.file_management

Help finding were to save output files and prevent overwriting files.

Author:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""

from os import mkdir
from pathlib import Path


def get_nonexistant_file(file):
    '''
    Given a base finds a file or directory name that does not exist.
    Arguments: 
        base (str):  the base to add indexes to, to ensure nonexistance
        suffix (str): a suffixes to the base+index to ensure nonexistance
    Returns: 
        Path: a file starting with base, ending with the suffix that 
            does not exist.
    '''
    file = Path(file)
    base = file.stem
    i = 1
    while Path(file).is_file():
        name = base + '_' + str(i)
        file = Path(file.parent, name+file.suffix)
        i += 1
    return Path(file)


def get_dir(base, overwrite=False, use_existing=False):
    '''
    Given a base finds a directory name that does not exist and make it.
    Can also choose to overwrite it
    Arguments: 
        base (str): the base of folder, may add indices to find non-existant
        overwrite (bool): if directory exists just overwrite it
        use_existing (bool): if directory exists don't overwrite
            or make a new one, just use existing
    Returns:
        Path: a directory starting with makes, the directory is already made
    '''
    name = base
    if overwrite:
        if Path(name).is_dir():
            print("WARNING: overwriting all information in", name)
            for sub in Path(name).iterdir():
                sub.unlink()
            Path(name).rmdir()
    elif use_existing:
        if Path(name).is_dir():
            return Path(name)
    else:
        i = 1
        while Path(name).is_dir():
            name = base + '_' + str(i)
            i += 1
    Path(name).mkdir()
    return Path(name)
