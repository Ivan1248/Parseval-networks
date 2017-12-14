from ioutils import path


def find_ancestor_sibling(path_end):
    try:
        return path.find_ancestor_sibling(__file__, path_end)
    except:
        print("ERROR: dirs.py: Could not find '"+path_end+"'.")
        return None


SAVED_MODELS = find_ancestor_sibling('data/models')
LOGS = find_ancestor_sibling('data/logs')
DATASETS = find_ancestor_sibling('projects/datasets')