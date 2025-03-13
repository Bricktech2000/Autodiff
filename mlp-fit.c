#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// // faster (90% accuracy)
// #define ETA 0.05   // learning rate
// #define BETA 0.9   // momentum coefficient
// #define LAMBDA 0.0 // regularization rate
// #define BATCH 100  // mini-batch size
// #define EPOCHS 500 // number of epochs

// slower (96% accuracy)
#define ETA 0.01     // learning rate
#define BETA 0.9     // momentum coefficient
#define LAMBDA 0.0   // regularization rate
#define BATCH 250    // mini-batch size
#define EPOCHS 10000 // number of epochs

#define TRAIN_OFST 16
#define TRAIN_LEN 60000
#define TRAIN_PATHS                                                            \
  "MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte"
#define TEST_OFST 8
#define TEST_LEN 10000
#define TEST_PATHS                                                             \
  "MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte"

#define ARRAY_FOR(ARRAY)                                                       \
  for (size_t idx = 0; idx < sizeof(ARRAY) / sizeof(*(ARRAY)); idx++)          \
    for (double elem = (ARRAY)[idx], *_p = &elem; _p;                          \
         (ARRAY)[idx] = elem, _p = NULL)

struct ex {
  x_t x;
  y_t y;
};

struct ex *load_mnist(char *x_path, char *y_path, size_t n) {
  FILE *x_fp = fopen(x_path, "r");
  if (x_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(x_fp, TRAIN_OFST, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  FILE *y_fp = fopen(y_path, "r");
  if (y_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(y_fp, TEST_OFST, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  struct ex *exs = malloc(sizeof(*exs) * n);

  int chr;
  for (size_t i = 0; i < n; i++) {
    struct ex *ex = exs + i;

    if ((chr = fgetc(y_fp)) == EOF)
      perror("fgetc"), exit(EXIT_FAILURE);
    if (chr >= sizeof(ex->y) / sizeof(*ex->y))
      abort();
    ARRAY_FOR(ex->y) elem = 0.0;
    ex->y[chr] = 1.0;

    ARRAY_FOR(ex->x) {
      if ((chr = fgetc(x_fp)) == EOF)
        perror("fgetc"), exit(EXIT_FAILURE);
      elem = (double)chr / 256.0;
    }
  }

  if (fclose(x_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  if (fclose(y_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  return exs;
}

int mnist_y_to_yi(y_t y) {
  int yi = 0;
  for (int i = 0; i < sizeof(y_t) / sizeof(*y); i++)
    yi = y[i] > y[yi] ? i : yi;
  return yi;
}

void mnist_x_dump(x_t x) {
  for (int j = 0; j < 28; j += 2) {
    for (int i = 0; i < 28; i++)
      putchar(" .,+*o%@"[(int)(x[28 * j + i] * 8.0)]);
    putchar('\n');
  }
}

void mnist_y_dump(y_t y) {
  for (int i = 0; i < sizeof(y_t) / sizeof(*y); i++)
    printf("%f ", y[i]);
  putchar('\n');
}

int main(void) {
  srand(time(NULL));

  struct ex *train_exs = load_mnist(TRAIN_PATHS, TRAIN_LEN);
  struct ex *test_exs = load_mnist(TEST_PATHS, TEST_LEN);

  static w_t w;
  static yh_t yh;
  static dw_t dw;
  c_t c = &(double){0.0};
  static dw_t v;

  ARRAY_FOR(v) elem = 0.0;
  ARRAY_FOR(w) elem = (double)rand() / RAND_MAX - 0.5;

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    ARRAY_FOR(dw) elem = 0.0;
    *c = 0.0;

    for (int batch = 0; batch < BATCH; batch++) {
      struct ex *ex = train_exs + rand() % TRAIN_LEN;
      mlp_backprop(ex->x, w, ex->y, dw, c);
    }

    *c /= BATCH;
    ARRAY_FOR(dw) elem /= BATCH;

    ARRAY_FOR(dw) elem += LAMBDA * w[idx] * w[idx];  // L2 regularization
    ARRAY_FOR(v) elem = elem * BETA - ETA * dw[idx]; // momentum
    ARRAY_FOR(w) elem += v[idx];                     // gradient descent

    printf("epoch %d of %d; loss %f", epoch, EPOCHS, *c);
    printf("%*s\n", (int)(*c * 64), "#");
  }

  double accuracy = 0.0;

  for (int i = 0; i < TEST_LEN; i++) {
    struct ex *ex = test_exs + i;
    mlp_predict(ex->x, w, yh);

    int correct = mnist_y_to_yi(yh) == mnist_y_to_yi(ex->y);
    accuracy += (double)correct / TEST_LEN;

    if (correct)
      continue;

    printf("yh  = "), mnist_y_dump(yh);
    printf("y   = "), mnist_y_dump(ex->y);
    printf("yhi = %d\n", mnist_y_to_yi(yh));
    printf("yi  = %d\n", mnist_y_to_yi(ex->y));
    mnist_x_dump(ex->x);
  }

  printf("MNIST accuracy: %f\n", accuracy);
}
