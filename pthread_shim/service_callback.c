#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
#endif

extern const char *const _libpthread_so_0_sym_names[];
extern void *_libpthread_so_0_tramp_table[];

// Callback that tries different library names
void *dlopen_callback(const char *lib_name) {
  void *lib_handle;
  lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);

  if(lib_handle == NULL) {
    printf("Failed to dlopen library");   
    exit(1);             
  }
  
  // dump the address of functions to file
  FILE *address_fp;
  address_fp = fopen("address.tmp", "w");
  size_t i = 0;
  const char * read;

  while ((read = _libpthread_so_0_sym_names[i]) != 0) {
    void *func_addr = dlsym(lib_handle, read);
    _libpthread_so_0_tramp_table[i] = func_addr;
    fprintf(address_fp, "%p\n", func_addr);
    i += 1;
  }

  printf("Function address is dumped into disk.\n");

  fclose(address_fp);

  return lib_handle;
}
