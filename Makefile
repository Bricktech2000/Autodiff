CC=gcc
CFLAGS=-O2 -Wall -Wextra -Wpedantic -std=c11

all: bin/mlp-fit bin/curve-fit bin/taylor

bin/taylor: taylor.c bin/autodiff.o | bin/
	$(CC) $(CFLAGS) $^ -lm -o $@

bin/curve-fit: curve-fit.c bin/tensor.o bin/autodiff.o utils.h | bin/
	$(CC) $(CFLAGS) -Wno-unused-function \
		curve-fit.c bin/tensor.o bin/autodiff.o -lm -o $@

bin/mlp-fit: mlp-fit.c bin/mlp-predict.o bin/mlp-backprop.o | bin/
	$(CC) $(CFLAGS) -Wno-unused-value -Wno-sign-compare -Ibin/ $^ -lm -o $@

bin/mlp-predict.o: bin/mlp-predict.c lib/runtime.h | bin/
	$(CC) $(CFLAGS) -O1 -Ilib/ -c $< -o $@
bin/mlp-backprop.o: bin/mlp-backprop.c lib/runtime.h | bin/
	$(CC) $(CFLAGS) -O1 -Ilib/ -c $< -o $@
bin/mlp-%.c: bin/mlp-gen | bin/
	cd bin/ && ./mlp-gen

bin/mlp-gen: mlp-gen.c bin/tensor.o bin/autodiff.o utils.h | bin/
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-value -Wno-missing-braces \
		mlp-gen.c bin/tensor.o bin/autodiff.o -lm -o $@

bin/tensor.o: lib/tensor.c lib/tensor.h lib/autodiff.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/autodiff.o: lib/autodiff.c lib/autodiff.h lib/runtime.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/:
	mkdir bin/

clean:
	rm -rf bin/
