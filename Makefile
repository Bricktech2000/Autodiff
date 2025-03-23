CC=gcc
CFLAGS=-O2 -Wall -Wextra -Wpedantic -std=c11

all: bin/curve-fit bin/mlp-fit

bin/curve-fit: curve-fit.c bin/utils.o bin/tensor.o bin/autodiff.o | bin/
	$(CC) $(CFLAGS) $^ -lm -o $@

bin/mlp-fit: mlp-fit.c bin/mlp-predict.o bin/mlp-backprop.o | bin/
	$(CC) $(CFLAGS) -Wno-sign-compare -Ibin/ $^ -lm -o $@

bin/mlp-%.o: bin/mlp-%.c lib/runtime.h | bin/
	$(CC) $(CFLAGS) -O1 -Ilib/ -c $< -o $@

bin/mlp-%.c: bin/mlp-gen | bin/
	cd bin/ && ./mlp-gen

bin/mlp-gen: mlp-gen.c bin/utils.o bin/tensor.o bin/autodiff.o | bin/
	$(CC) $(CFLAGS) -Wno-unused-value $^ -lm -o $@

bin/utils.o: lib/utils.c lib/utils.h lib/tensor.h lib/autodiff.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/tensor.o: lib/tensor.c lib/tensor.h lib/autodiff.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/autodiff.o: lib/autodiff.c lib/autodiff.h lib/runtime.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/:
	mkdir bin/

clean:
	rm -rf bin/
