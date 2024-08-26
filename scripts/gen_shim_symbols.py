import argparse
import os

me = os.path.basename(__file__)

def parse_exe_sym(exe_path):
    symbol_names = {}

    out_file = "exe_syms.tmp"
    command_str = f"objdump -T {exe_path} > {out_file}"
    os.system(command_str)
    syms = None
    with open(out_file, "r") as f:
        syms = f.read()
    
    for line in syms.split("\n"):
        values = line.split()
        
        offset = 0

        # if w is present
        if len(values) == 7:
            offset = 1
        if len(values) < 6 or len(values) > 7:
            continue

        is_undefined = values[offset + 2] == "*UND*"
        if is_undefined:
            # TODO: handle more types like DO
            is_dynamic_function = values[offset + 1] == "DF"
            if not is_dynamic_function:
                continue

            symbol_names[values[offset + 5]] = True
    
    return symbol_names

def search_lib_sym(lib_path, symbol_names):
    result = set()
    out_file = "lib_syms.tmp"
    command_str = f"objdump -T {lib_path} > {out_file}"
    os.system(command_str)
    syms = None
    with open(out_file, "r") as f:
        syms = f.read()
    
    for line in syms.split("\n"):
        values = line.split()
        
        offset = 0

        # if w is present
        if len(values) == 7:
            offset = 1
        if len(values) < 6 or len(values) > 7:
            continue

        is_text = values[offset + 2] == ".text"
        
        if is_text:
            # TODO: handle more types like DO
            is_dynamic_function = values[offset + 1] == "DF"
            if not is_dynamic_function:
                continue

            symbol_name = values[offset + 5]
            if symbol_name in symbol_names:
                result.add(symbol_name)
    
    return result
            
def main():
    parser = argparse.ArgumentParser(description="Generate wrappers for shared library functions.",
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog=f"""\
    Examples:
    $ python3 {me} hello_world /usr/lib/x86_64-linux-gnu/lib.so.0
    """)

    parser.add_argument('executable',
                        metavar='EXE',
                        help="Executable which library linked to.")
    parser.add_argument('library',
                        metavar='LIB',
                        help="Library to be wrapped.")
    args = parser.parse_args()

    exe_path = args.executable
    lib_path = args.library

    symbol_names = parse_exe_sym(exe_path)
    result = search_lib_sym(lib_path, symbol_names)
    for r in result:
        print(r)


if __name__ == "__main__":
    main()