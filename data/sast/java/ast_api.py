
from util.helper import traverse_src_files
import os
from util.setting import log
from data.sast.java.src_parser import java_parser



def create_ast(file_path: str) -> None:
    
    all_files = traverse_src_files(os.path.join(file_path), 'java')
    func_list = list()
    log.info('Extract functions...')

    for file in all_files:
        file_func = java_parser(file)
        log.info(f'Parsing file: {file}')
        if not file_func.has_type('ERROR'):
            func_list.append(file_func)
        else:
            log.error(f'File: {file} \t function: {func.func_name} has ERROR Type.')
            exit(-1)

    return func_list           


        


