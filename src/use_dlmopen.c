#define _GNU_SOURCE

#include <dlfcn.h>
#include <string.h>
#include <stdio.h>

typedef void *(*orig_dlopen_t)(const char *filename, int flags);
orig_dlopen_t orig_dlopen;

void *cuda_ld;

int cuda_count = 0;

void *dlopen(const char *filename, int flags) {
    if (filename && strcmp(filename, "libcuda.so.1") == 0) {
        printf("dlmopen: %s\n", filename);
        fflush(stdout);
        cuda_count++;
        if (cuda_count % 2 == 0)
            return cuda_ld;
        else
            cuda_ld = dlmopen(LM_ID_NEWLM, filename, flags);
        return cuda_ld;
    } else {
        printf("dlopen: %s\n", filename);
        fflush(stdout);
        return orig_dlopen(filename, flags);
    }
}

__attribute__((constructor)) static void setup(void) {
  orig_dlopen = dlsym(RTLD_NEXT, "dlopen"); 
  fprintf(stderr, "called setup()\n");
}