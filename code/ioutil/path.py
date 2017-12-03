import os.path


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path)


def get_file_name_without_extension(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]

def find_ancestor(path, ancestor_name):
    components = os.path.abspath(path).split(os.sep)
    return str.join(os.sep, components[:components.index(ancestor_name)+1])