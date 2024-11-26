include ../make.def

.c.o:
	${CC} $(CFLAGS) -c $*.c -fPIC

ARRAYOBJS = arraybench.o common.o

ARRAYOBJS_VLC = arraybench_vlc.o common.o

ARRAYOBJS_SHARE = arraybench_share.o common.o

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
arraybench_ida.h: arraybench.h
	$(CC) -DIDA=$(IDA) -E arraybench.h -o $@

# Multiple object files due to multiple array sizes, makes object file arraybench_*.o
arraybench.o:arraybench_ida.h arraybench.c
	$(CC) $(CFLAGS) -DIDA=$(IDA) arraybench.c -c -o $@

arraybench_share.o:arraybench_ida.h arraybench_share.c
	$(CC) $(CFLAGS) -DIDA=$(IDA) arraybench_share.c -c -o $@ -fPIC

arraybench_vlc.o: arraybench_ida.h arraybench_transparent.cpp
	$(CC) $(CFLAGS) -DIDA=$(IDA) $(C_INC) arraybench_transparent.cpp -c -o $@

# Multiple executables due to multiple array sizes, makes exe file arraybench_*
arraybench: $(ARRAYOBJS) arraybench.c
	$(CC) $(CFLAGS) $(ARRAYOBJS) -o $@

arraybench_transparent: $(ARRAYOBJS_VLC) arraybench_transparent.cpp
	$(CC) $(CFLAGS) $(ARRAYOBJS_VLC) -o $@

arraybench.so: $(ARRAYOBJS_SHARE) arraybench_share.c
	$(CC) $(CFLAGS) $(ARRAYOBJS_SHARE) -shared -o $@

clean:
	rm -f arraybench_ida.h arraybench_vlc_$(IDA).* common.o arraybench arraybench_share.o arraybench.o


