from typing import List
import os.path


def get_files(directory: str) -> List[str]:
    """ Returns a list of full paths of files in the directory. """
    return [f for f in (os.path.join(directory, e) for e in os.listdir(directory)) if os.path.isfile(f)]

