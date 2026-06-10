.POSIX:
.SUFFIXES:
CC=gcc
CFLAGS=-O2 -Wall -Wextra -Wpedantic -std=c11 -lm

all: bin/mlp-fit bin/mlp-gen bin/curve-fit bin/taylor
bin/:; mkdir bin/
clean:; rm -rf bin/

bin/taylor:    bin/autodiff.o taylor.c;                         $(CC) $(CFLAGS) -o $@ bin/autodiff.o taylor.c
bin/curve-fit: bin/autodiff.o bin/tensor.o utils.h curve-fit.c; $(CC) $(CFLAGS) -o $@ bin/autodiff.o bin/tensor.o curve-fit.c -Wno-unused-function
bin/mlp-gen:   bin/autodiff.o bin/tensor.o utils.h mlp-gen.c;   $(CC) $(CFLAGS) -o $@ bin/autodiff.o bin/tensor.o mlp-gen.c -Wno-unused-function -Wno-unused-value -Wno-missing-braces
bin/mlp-fit:   bin/mlp-predict.o bin/mlp-backprop.o mlp-fit.c;  $(CC) $(CFLAGS) -o $@ bin/mlp-predict.o bin/mlp-backprop.o -Ibin/ mlp-fit.c -Wno-unused-value -Wno-sign-compare

bin/mlp-predict.o:  lib/runtime.h bin/mlp.h bin/mlp-predict.c;  $(CC) $(CFLAGS) -o $@ -O1 -Ilib/ -c bin/mlp-predict.c
bin/mlp-backprop.o: lib/runtime.h bin/mlp.h bin/mlp-backprop.c; $(CC) $(CFLAGS) -o $@ -O1 -Ilib/ -c bin/mlp-backprop.c
bin/mlp-predict.c bin/mlp-backprop.c bin/mlp.h: bin/mlp-stamp
bin/mlp-stamp: bin/mlp-gen; cd bin/ && ./mlp-gen && touch mlp-stamp

bin/tensor.o:   bin/ lib/autodiff.h lib/tensor.h lib/tensor.c;    $(CC) $(CFLAGS) -o $@ -c lib/tensor.c -Wno-parentheses -Wno-missing-field-initializers
bin/autodiff.o: bin/ lib/autodiff.h lib/runtime.h lib/autodiff.c; $(CC) $(CFLAGS) -o $@ -c lib/autodiff.c
