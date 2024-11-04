
import os


def traverse_files(dir_path: str = None) -> list:
    all_files = list()
    walk_tree = os.walk(dir_path)
    for root, _, files in walk_tree:
        for file in files:
            all_files.append(os.path.join(root, file))
    
    return all_files

def check_extension(file_name: str, extension: str) -> bool:
    _extension = os.path.splitext(file_name)[-1][1:]
    if _extension == extension:
        return True
    return False

def traverse_src_files(dir_path: str, extension: str) -> list:
    """Obtain all source files we want to parse.

    attributes:
        dir_path -- the directory path we want to parse.
        extension -- the file extension we want to parse (e.g., 'java')
    
    returns:
        files -- list including files we want to parse.
    """
    files = list()
    all_files = traverse_files(dir_path)
    for file in all_files:
        if check_extension(file, extension):
            files.append(file)
    
    return files