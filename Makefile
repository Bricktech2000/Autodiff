CC=gcc
CFLAGS=-O2 -Wall -Wextra -Wpedantic -std=c99

all: bin/mlp-fit

bin/mlp-fit: mlp-fit.c bin/mlp.o
	mkdir -p bin/
	$(CC) $(CFLAGS) -lm $^ -o $@

bin/mlp.o: bin/mlp.c
	mkdir -p bin/
	$(CC) $(CFLAGS) -c $^ -o $@

bin/mlp.c: mlp-gen.c autodiff.c
	mkdir -p bin/
	$(CC) $(CFLAGS) -Wno-unused-parameter $< -o bin/mlp-gen
	bin/mlp-gen $@

clean:
	rm -rf bin/
