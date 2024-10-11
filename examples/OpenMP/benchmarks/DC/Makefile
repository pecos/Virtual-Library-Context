SHELL=/bin/sh
BENCHMARK=dc
BENCHMARKU=DC

include ../make.def

include ../make.common

OBJS = adc.o dc.o extbuild.o rbt.o jobcntl.o \
	${COMMON}/c_print_results.o  \
	${COMMON}/c_timers.o ${COMMON}/c_wtime.o

VLC_OBJS = adc.o dc_transparent.o extbuild.o rbt.o jobcntl.o \
	${COMMON}/c_print_results.o  \
	${COMMON}/c_timers.o ${COMMON}/c_wtime.o

dc: ${OBJS} 
	${CLINK} ${CLINKFLAGS} -o $@ ${OBJS} ${C_LIB}

dc_transparent: ${VLC_OBJS} 
	${CLINK} ${CLINKFLAGS} -o $@ ${VLC_OBJS} ${C_LIB}

adc.o: adc.c npbparams.h
	${CCOMPILE} $<

dc.o: dc.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $<

dc_transparent.o: dc_transparent.cpp adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $<

extbuild.o: extbuild.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $<

rbt.o: rbt.c adcc.h adc.h rbt.h macrodef.h npbparams.h
	${CCOMPILE} $<

jobcntl.o: jobcntl.c adcc.h adc.h macrodef.h npbparams.h
	${CCOMPILE} $<

clean:
	- rm -f *.o 
	- rm -f core
	- rm -f ADC.logf.* ADC.view.* 

