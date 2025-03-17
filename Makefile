CC=gcc
CFLAGS=-O2 -Wall -Wextra -Wpedantic -std=c99

all: bin/curve-fit bin/mlp-fit

bin/curve-fit: curve-fit.c bin/utils.o bin/tensor.o bin/autodiff.o | bin/
	$(CC) $(CFLAGS) -lm $^ -o $@

bin/mlp-fit: mlp-fit.c bin/mlp.o | bin/
	$(CC) $(CFLAGS) -Wno-sign-compare -Ibin/ -lm $^ -o $@

bin/mlp.o: bin/mlp.c | bin/
	$(CC) $(CFLAGS) -O1 -c $^ -o $@

bin/mlp.c: mlp-gen.c bin/utils.o bin/tensor.o bin/autodiff.o | bin/
	$(CC) $(CFLAGS) -Wno-unused-value -lm $^ -o bin/mlp-gen
	cd bin/ && ./mlp-gen

bin/utils.o: lib/utils.c lib/utils.h lib/tensor.h lib/autodiff.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/tensor.o: lib/tensor.c lib/tensor.h lib/autodiff.c | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/autodiff.o: lib/autodiff.c lib/autodiff.h | bin/
	$(CC) $(CFLAGS) -c $< -o $@

bin/:
	mkdir bin/

clean:
	rm -rf bin/
