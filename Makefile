OPENMP=0
NNPACK=1
ARM_NEON=1
DEBUG=0

VPATH=./src
DISTRIOT=./distriot
DARKNET=./darknet-nnpack
OBJDIR=./obj/
EXEC=deepthings
DARKNETLIB=libdarknet.a
DISTRIOTLIB=libdistriot.a

NNPACK=/home/sky/NNPACK/build
CPUINFO=/home/sky/DeepThings/cpuinfo/build

CC=gcc
LDFLAGS= -lm -pthread
CFLAGS=-Wall -fPIC
COMMON=-I$(DISTRIOT)/include/ -I$(DISTRIOT)/src/ -I$(DARKNET)/include/ -I$(DARKNET)/src/ -Iinclude/ -Isrc/ 
LDLIB=-L$(DISTRIOT) -l:$(DISTRIOTLIB) -L$(DARKNET) -l:$(DARKNETLIB)

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS+=-O0 -g
else
OPTS+=-Ofast
endif

ifeq ($(NNPACK), 1)
COMMON+= -DNNPACK
CFLAGS+= -DNNPACK
LDFLAGS+= -lnnpack -lpthreadpool
endif

ifeq ($(ARM_NEON), 1)
COMMON+= -DARM_NEON
CFLAGS+= -DARM_NEON   -funsafe-math-optimizations -ftree-vectorize
endif

CFLAGS+=$(OPTS)
OBJS = top.o ftp.o inference_engine_helper.o frame_partitioner.o adjacent_reuse_data_serialization.o self_reuse_data_serialization.o deepthings_edge.o deepthings_gateway.o cmd_line_parser.o
EXECOBJ = $(addprefix $(OBJDIR), $(OBJS))
DEPS = $(wildcard */*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(EXECOBJ) $(DARKNETLIB) $(DISTRIOTLIB)
	$(CC) $(COMMON) $(CFLAGS) $(EXECOBJ) -o $@  $(LDLIB) $(LDFLAGS)

$(DARKNETLIB):
	$(MAKE) -C $(DARKNET) NNPACK=$(NNPACK) ARM_NEON=$(ARM_NEON) DEBUG=$(DEBUG) OPENMP=$(OPENMP) all

$(DISTRIOTLIB):
	$(MAKE) -C $(DISTRIOT) DEBUG=$(DEBUG) all

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

test:
	./deepthings ${ARGS}

.PHONY: clean clean_all

clean_all:
	make -C $(DISTRIOT) clean
	make -C $(DARKNET) clean
	rm -rf $(EXEC) $(EXECOBJ) *.log $(OBJDIR) *.png 

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log $(OBJDIR) *.png 
