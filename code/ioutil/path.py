import os.path


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path)


def get_file_name_without_extension(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def find_ancestor(path, ancestor_name):
    components = os.path.abspath(path).split(os.sep)
    return os.path.normpath(
        str.join(os.sep, components[:components.index(ancestor_name) + 1]))


def find_ancestor_sibling(path, ancestor_sibling_name):
    """ 
    ancestor_sibling_name can be the name of a sibling directory to some
    ancestor, but it can be a descendant of the sibling as well 
    """
    components = os.path.abspath(path).split(os.sep)
    while len(components) > 0:
        path = os.path.normpath(
            str.join(os.sep, components + [ancestor_sibling_name]))
        if os.path.isdir(path):
            return path
        components.pop()
    assert False, "No ancestor sibling found"
