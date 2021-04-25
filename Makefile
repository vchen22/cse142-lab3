default: benchmark.csv run_tests.exe regressions.out code.csv  
OPTIMIZE+=-march=x86-64
CLEANUP=trace_traceme.hdf5 trace_code.hdf5
include $(ARCHLAB_ROOT)/cse141.make
$(BUILD)code.s: $(BUILD)opt_cnn.hpp

MEMOPS?=5000000

ifeq ($(DEVEL_MODE),yes)
OUR_CMD_LINE_ARGS=--stat runtime=ARCHLAB_WALL_TIME 
else
OUR_CMD_LINE_ARGS=--stat-set L2.cfg
endif

FULL_CMD_LINE_ARGS=$(LAB_COMMAND_LINE_ARGS) $(CMD_LINE_ARGS)

code.csv: code.exe
	rm -f gmon.out
	./code.exe --stats-file $@ $(FULL_CMD_LINE_ARGS)
	pretty-csv $@
	if [ -e gmon.out ]; then gprof $< > code.gprof; fi

traceme_trace: traceme_trace.hdf5
traceme_trace.hdf5: traceme.exe
	mtrace --trace traceme --main aoeu --memops $(MEMOPS) -- ./traceme.exe

traceme.exe: traceme.cpp
	g++ $(USER_CFLAGS) $< -o $@

code_trace: code_trace.hdf5
code_trace.hdf5: code.exe
	mtrace --trace code --main none --memops $(MEMOPS)  --  ./code.exe --stats-file $@ $(FULL_CMD_LINE_ARGS)

.PHONY: regressions.out
regressions.out: ./run_tests.exe
	-./run_tests.exe > $@ 
	tail -1 $@

# We run the same test again but without their command line argument.
# A better solution might be to somehow lock down --dataset and
# --scale, but that'd require a lot of carefuly checking.
benchmark.csv: code.exe
	rm -f gmon.out
	                                              #--scale 4 --reps 200 --train-reps 1000
	./code.exe --stats-file $@ --dataset cifar100 --scale 4 --reps 500 --train-reps 3000 $(OUR_CMD_LINE_ARGS)
	pretty-csv $@
	if [ -e gmon.out ]; then gprof $< > benchmark.gprof; fi

