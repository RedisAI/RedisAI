#!/usr/bin/python
# generates the llapi docs based on the doxygen comments on the header files
# Python 3.X
# Version 0.1

import argparse
import os
import re
import subprocess
import sys


def whereis(program):
    for path in os.environ.get('PATH', '').split(':'):
        if os.path.exists(os.path.join(path, program)) and \
                not os.path.isdir(os.path.join(path, program)):
            return os.path.join(path, program)
    return None


# Check if system has the required utilities, cset numactl etc
def required_utilities(utility_list):
    required_utilities = 1
    for index in utility_list:
        if whereis(index) == None:
            print('Cannot locate ' + index + ' in path!')
            required_utilities = 0
    return required_utilities


def extract_public_functions_documentation(data, md_function_def_map):
    first_split = data.split("## Public Functions Documentation")
    if len(first_split) == 2:
        public_functions_definition = first_split[1]
        functions_split = public_functions_definition.split("### function")
        for function_md_block in functions_split:
            function_md_block_lines = function_md_block.splitlines()
            func = function_md_block_lines[0].strip()
            md_function_def_map[func] = function_md_block_lines[1:]


def prune_and_merge_markdown(llapi_functions, md_function_def_map):
    final_markdown = []
    for llapi_function in llapi_functions:
        internal_naming = "RAI\\_{}".format(llapi_function)
        llapi_function_name = "RedisAI\\_{}".format(llapi_function)
        if internal_naming in md_function_def_map:
            final_markdown.append("### {}".format(llapi_function_name))
            for line in md_function_def_map[internal_naming]:
                # clear links
                p = re.compile(r'\[(\*\*\S+\*\*)\]\(\S+\)')
                newline = p.sub(r'\1', line)
                
                # move from cpp to c code snippets
                if newline.startswith("```cpp"):
                    newline = newline.replace("```cpp", "```c")

                    # change from RAI_* to RedisAI_* in the code snipets and comments
                p = re.compile(r'RAI\_(\w+)')
                newline = p.sub(r'RedisAI_\1', newline)
                final_markdown.append(newline)

    return final_markdown


def extra_llapi_functions(register_src_file):
    llapi_functions = []
    with open(register_src_file, "r") as mainredis:
        for line in mainredis.readlines():
            fname_regex = re.search(
                '.*REGISTER_API\((\w+),.*\)', line)
            if fname_regex is not None:
                llname_sufix = fname_regex.group(1)
                llapi_functions.append(llname_sufix)

    llapi_functions.sort()
    return llapi_functions


def generate_md_function_def_map(llapi_mkdown_dir):
    md_function_def_map = {}
    for f in files:
        if "8h.md" in f:
            with open("{}/{}".format(llapi_mkdown_dir, f), 'r') as file_content:
                data = file_content.read()
                extract_public_functions_documentation(data, md_function_def_map)

    return md_function_def_map

def run_doxigen():
    cp = subprocess.Popen(['doxygen'], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    cp.communicate()
    rc = cp.returncode
    return True if rc == 0 else False

def clean_doxigen():
    cp = subprocess.Popen(['rm -rf xml'], shell=True, stdout=subprocess.PIPE)
    cp.communicate()
    rc = cp.returncode
    return  True if rc == 0 else False


def run_doxybook(input,output):
    cp = subprocess.Popen(['doxybook -t mkdocs -i {} -o {}'.format(input,output)], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    cp.communicate()
    rc = cp.returncode
    return  True if rc == 0 else False

def clean_doxybook(doxybook_folder):
    cp = subprocess.Popen(['rm -rf {}'.format(doxybook_folder)], shell=True, stdout=subprocess.PIPE)
    cp.communicate()
    rc = cp.returncode
    return  True if rc == 0 else False

# Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generates the llapi docs based on the doxygen comments on the header files.')
    parser.add_argument('--register-src-file', type=str, default="./../src/redisai.c",
                        help='file including the call to REGISTER_API macro')
    parser.add_argument('--out', type=str, default="api_reference.md", help='file to save the final markdown')
    args = parser.parse_args()

    if required_utilities(['doxygen', 'doxybook']) == 0:
        print('Utilities Missing! Exiting..')
        sys.exit(1)

    doxybook_temp_dir = "doxybook_temp"

    print('Running doxygen step...')
    result = run_doxigen()
    if result == 0:
        print('Something went wrong on doxygen step! Exiting..')
        sys.exit(1)

    print('Running doxybook step...')
    result = run_doxybook("xml",doxybook_temp_dir)
    if result == 0:
        print('Something went wrong on doxybook step! Exiting..')
        sys.exit(1)

    print('Searching on {} for REGISTER_API macro...'.format(args.register_src_file))
    llapi_functions = extra_llapi_functions(args.register_src_file)

    files = os.listdir(doxybook_temp_dir)
    files.sort()
    md_function_def_map = generate_md_function_def_map(doxybook_temp_dir)
    print('Associating {} registered llapi functions with their code comments...'.format(len(llapi_functions)))   
    final_markdown = prune_and_merge_markdown(llapi_functions, md_function_def_map)

    print('Saving markdown to {}...'.format(args.out))
    with open(args.out, "w") as output_md:
        output_md.write("# RedisAI low-level API\n\n"
                        "The RedisAI low-level API makes RedisAI available as a library that can be used by other Redis modules written in C or Rust.\n"
                        "Other modules will be able to use this API by calling the function RedisModule_GetSharedAPI() and casting the return value to the right function pointer.\n\n"
                        )

        output_md.write("## Public Functions Documentation\n\n")
        for line in final_markdown:
            output_md.write(line + "\n")

    clean_doxigen()
    clean_doxybook(doxybook_temp_dir)
