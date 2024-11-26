SHELL=/bin/sh
BENCHMARK=dc
BENCHMARKU=DC

include ../make.def

include ../make.common

OBJS = adc.o dc.o extbuild.o rbt.o jobcntl.o \
	${COMMON}/c_print_results.o  \
	${COMMON}/c_timers.o ${COMMON}/c_wtime.o

OBJS_VLC = adc.o dc_transparent.o extbuild.o rbt.o jobcntl.o \
	${COMMON}/c_print_results.o  \
	${COMMON}/c_timers.o ${COMMON}/c_wtime.o

OBJS_SHARED = adc.o dc_shared.o extbuild.o rbt.o jobcntl.o \
	${COMMON}/c_print_results.o  \
	${COMMON}/c_timers.o ${COMMON}/c_wtime.o

dc: ${OBJS} 
	${CLINK} ${CLINKFLAGS} -o $@ ${OBJS} ${C_LIB}

dc.so: ${OBJS_SHARED} 
	${CLINK} ${CLINKFLAGS} -shared -o $@ ${OBJS_SHARED} ${C_LIB}

dc_transparent: ${OBJS_VLC} 
	${CLINK} ${CLINKFLAGS} -o $@ ${OBJS_VLC} ${C_LIB}

adc.o: adc.c npbparams.h
	${CCOMPILE} $< -fPIC

dc.o: dc.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $<

dc_transparent.o: dc_transparent.cpp adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $<

dc_shared.o: dc_shared.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $< -fPIC

extbuild.o: extbuild.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $< -fPIC

rbt.o: rbt.c adcc.h adc.h rbt.h macrodef.h npbparams.h
	${CCOMPILE} $< -fPIC

jobcntl.o: jobcntl.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $< -fPIC

clean:
	- rm -f *.o 
	- rm -f core
	- rm -f ADC.logf.* ADC.view.*
	- rm -f dc dc_transparent
	- rm -f ../common/*.o
