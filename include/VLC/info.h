#ifndef _VLC_INFO_H_
#define _VLC_INFO_H_

#ifdef NDEBUG
#define VLC_DEBUG(...) 
#else
#define VLC_DEBUG(...) ({\
            printf("[DEBUG] ");\
            printf(__VA_ARGS__);\
            printf("\n");\
           })
#endif

#define VLC_DIE(...) ({\
            printf("[ERROR] ");\
            printf(__VA_ARGS__);\
            printf("\n");\
            std::exit(EXIT_FAILURE);\
           })

#endif // _VLC_INFO_H_