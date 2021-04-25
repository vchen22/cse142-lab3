CMD_LINE_ARGS=--engine papi --stat INSTRUCTIONS_RETIRED --stat BRANCH_INSTRUCTIONS_RETIRED --mat-small 96 --mat-large 768 --iterations 10
include $(ARCHLAB_ROOT)/compile.make

%.gprof: %.exe gmon.out
	prof %.exe > %.gprof

.PHONY: run-submission
run-submission: default

%.exe : %.o ../lab_files/main.o
	$(CXX) $(LDFLAGS) $^ -o $@

