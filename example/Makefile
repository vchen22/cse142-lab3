ifndef ARCHLAB_ROOT
$(error ARCHLAB_ROOT is not set.  Have you sourced config.sh?  Maybe do it again.)
endif
ifndef PIN_ROOT
$(error PIN_ROOT is not set.  Have you sourced config.sh?  Maybe do it again.)
endif
ifndef CANELA_ROOT
$(error CANELA_ROOT is not set.  Have you sourced config.sh?  Maybe do it again.)
endif

OPTIMIZE=-O3 $(EXTRA)

default:stabilize.exe

include $(ARCHLAB_ROOT)/cse141.make

DATASET=cifar

.PHONY: traces
traces:
	for i in baseline reorder_pixelxy tile_y_1 pretile_y; do  echo mtrace --trace $${i}_$(DATASET) --cache-line-count 128 -- ./stabilize.exe --dataset $(DATASET) --impl $$i;done

