#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"
#endif

extern const char *const sym_names[];
extern void *_libfoobar_so_tramp_table[];

void *dlopen_callback(const char *lib_name, int symbol_offset, int vlc_id) {
    void *lib_handle = NULL;
    char buf[128];
    sprintf(buf, "address.%d", vlc_id);
    // if the file exists, it means this shim is already loaded once, so simply read from the address file
    if (access(buf, F_OK) == 0) {
        printf("already loaded once.\n");
        // load the address of functions from file
        FILE *address_fp;
        address_fp = fopen(buf, "r");
        size_t i = 0;
        char * line = NULL;
        size_t len = 0;
        size_t read;

        while ((read = getline(&line, &len, address_fp)) != -1) {
            void *func_addr = (void *) strtoul(line, NULL, 0);
            _libfoobar_so_tramp_table[i + symbol_offset] = func_addr;
            i += 1;
        }

        for (int i = 0; i < 7; i++) {
            printf("debug: %d - %ld \n", i,  (unsigned long) _libfoobar_so_tramp_table[i]);
        }

        fclose(address_fp);
        free(line);

        printf("Function address is loaded from disk.\n");
    } else { // not loaded before
        lib_handle = dlopen(lib_name, RTLD_NOW);

        if(lib_handle == NULL) {
            printf("Failed to dlopen library\n");   
            exit(1);             
        }
        
        // dump the address of functions to file
        size_t i = 0;
        const char * read;

        while ((read = sym_names[i]) != 0) {
            void *func_addr = dlsym(lib_handle, read);
            // the first copy should have symbol_offset=0 in the table by default
            _libfoobar_so_tramp_table[i] = func_addr;
            i += 1;
        }

        printf("first copy is created.\n");

    }

    // lib hanlder will not be used actually
    return lib_handle;
}
