include ../make.def

.c.o:
	${CC} $(CFLAGS) -c $*.c

ARRAYOBJS = arraybench_$(IDA).o common.o

ARRAYOBJS_VLC = arraybench_vlc_$(IDA).o common.o

# all:
# 	$(MAKE) IDA=1 arraybench
# 	$(MAKE) IDA=3 arraybench
# 	$(MAKE) IDA=9 arraybench
# 	$(MAKE) IDA=27 arraybench
# 	$(MAKE) IDA=81 arraybench
# 	$(MAKE) IDA=243 arraybench
# 	$(MAKE) IDA=729 arraybench
# 	$(MAKE) IDA=2187 arraybench
# 	$(MAKE) IDA=6561 arraybench
# 	$(MAKE) IDA=19683 arraybench
# 	$(MAKE) IDA=59049 arraybench

# Multiple header files due to multiple array sizes, makes header file arraybench_*.h
arraybench_$(IDA).h: arraybench.h
	$(CC) -DIDA=$(IDA) -E arraybench.h -o $@

# Multiple object files due to multiple array sizes, makes object file arraybench_*.o
arraybench_$(IDA).o: arraybench_$(IDA).h arraybench.c
	$(CC) $(CFLAGS) -DIDA=$(IDA) arraybench.c -c -o $@

arraybench_vlc_$(IDA).o: arraybench_$(IDA).h arraybench_transparent.cpp
	$(CC) $(CFLAGS) -DIDA=$(IDA) $(C_INC) arraybench_transparent.cpp -c -o $@

# Multiple executables due to multiple array sizes, makes exe file arraybench_*
arraybench: $(ARRAYOBJS) arraybench.c
	$(CC) $(CFLAGS) $(ARRAYOBJS) -o $@

arraybench_transparent: $(ARRAYOBJS_VLC) arraybench.c
	$(CC) $(CFLAGS) $(ARRAYOBJS_VLC) -o $@

clean:
	rm -f arraybench_$(IDA).* arraybench_vlc_$(IDA).* common.o arraybench


