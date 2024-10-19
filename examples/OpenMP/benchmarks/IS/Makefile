SHELL=/bin/sh
BENCHMARK=is
BENCHMARKU=IS

include ../make.def

include ../make.common

OBJS = is.o \
       ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o

OBJS_VLC = is_transparent.o \
       ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o

is: ${OBJS}
	${CLINK} ${CLINKFLAGS} -o $@ ${OBJS} ${C_LIB}

is_transparent: ${OBJS_VLC}
	${CLINK} ${CLINKFLAGS} -o $@ ${OBJS_VLC} ${C_LIB}

is.o:   is.cpp  npbparams.h
	${CCOMPILE} $<

is_transparent.o:   is_transparent.cpp  npbparams.h
	${CCOMPILE} $<


clean:
	rm -f *.o is_transparent is
