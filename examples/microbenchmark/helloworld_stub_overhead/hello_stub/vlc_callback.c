#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"
#endif

extern const char *const sym_names[];
extern void *_libhello_so_tramp_table[];

void *dlopen_callback(const char *lib_name) {
    void *lib_handle = NULL;
    // if the file exists, it means this shim is already loaded once, so simply read from the address file
    if (access("address.tmp", F_OK) == 0) {
        printf("already loaded once.\n");
        // load the address of functions from file
        FILE *address_fp;
        address_fp = fopen("address.tmp", "r");
        size_t i = 0;
        char * line = NULL;
        size_t len = 0;
        size_t read;

        while ((read = getline(&line, &len, address_fp)) != -1) {
            void *func_addr = (void *) strtol(line, NULL, 0);
            _libhello_so_tramp_table[i] = func_addr;
            i += 1;
        }

        fclose(address_fp);
        free(line);

        printf("Function address is loaded from disk.\n");
    } else { // not loaded before
        lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);

        if(lib_handle == NULL) {
            printf("Failed to dlopen library\n");   
            exit(1);             
        }
        
        // dump the address of functions to file
        FILE *address_fp;
        address_fp = fopen("address.tmp", "w");
        size_t i = 0;
        const char * read;

        while ((read = sym_names[i]) != 0) {
            void *func_addr = dlsym(lib_handle, read);
            _libhello_so_tramp_table[i] = func_addr;
            fprintf(address_fp, "%p\n", func_addr);
            i += 1;
        }

        printf("Function address is dumped into disk.\n");

        fclose(address_fp);
    }

    // lib hanlder will not be used actually
    return lib_handle;
}
