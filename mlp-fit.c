#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// // faster
// #define ETA 0.05    // learning rate
// #define BETA 0.9    // momentum coefficient
// #define LAMBDA 0.0  // regularization rate
// #define BATCH 100   // mini-batch size
// #define EPOCHS 1000 // number of epochs

// slower
#define ETA 0.01     // learning rate
#define BETA 0.9     // momentum coefficient
#define LAMBDA 0.0   // regularization rate
#define BATCH 250    // mini-batch size
#define EPOCHS 10000 // number of epochs

#define ARRAY_FOR(ARRAY)                                                       \
  for (size_t idx = 0; idx < sizeof(ARRAY) / sizeof(*ARRAY); idx++)            \
    for (double elem = ARRAY[idx], *e = &elem; e; ARRAY[idx] = elem, e = NULL)

int main(void) {
  srand(time(NULL));

  FILE *x_fp = fopen("MNIST/train-images.idx3-ubyte", "r");
  if (x_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(x_fp, 16, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  FILE *y_fp = fopen("MNIST/train-labels.idx1-ubyte", "r");
  if (y_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(y_fp, 8, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  static x_t x;
  static w_t w;
  static yh_t yh;
  static y_t y;
  static dw_t dw;
  c_t c = &(double){0.0};
  static dw_t v;

  ARRAY_FOR(v) elem = 0.0;
  ARRAY_FOR(w) elem = (double)rand() / RAND_MAX - 0.5;

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    ARRAY_FOR(dw) elem = 0.0;
    *c = 0.0;

    for (int batch = 0; batch < BATCH; batch++) {
      ARRAY_FOR(x) elem = (double)fgetc(x_fp) / 256;
      ARRAY_FOR(y) elem = 0.0;
      int yi = fgetc(y_fp); // XXX bounds check
      y[yi] = 1.0;

      backprop(x, w, y, dw, c);
    }

    *c /= BATCH;
    ARRAY_FOR(dw) elem /= BATCH;

    ARRAY_FOR(dw) elem += LAMBDA * w[idx] * w[idx];  // L2 regularization
    ARRAY_FOR(v) elem = elem * BETA - ETA * dw[idx]; // momentum
    ARRAY_FOR(w) elem += v[idx];                     // gradient descent

    printf("epoch %d of %d; loss %f %*.s@\n", epoch, EPOCHS, *c, (int)(*c * 64),
           "");

    // XXX hacky as hell
    if (ftell(y_fp) >= 50000) {
      int ofst = rand() % 128;
      fseek(x_fp, 16 + 784 * ofst, SEEK_SET);
      fseek(y_fp, 8 + ofst, SEEK_SET);
    }
  }

  if (fclose(x_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  if (fclose(y_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  // x_fp = fopen("MNIST/train-images.idx3-ubyte", "r");
  x_fp = fopen("MNIST/t10k-images.idx3-ubyte", "r");
  if (x_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(x_fp, 16, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  // y_fp = fopen("MNIST/train-labels.idx1-ubyte", "r");
  y_fp = fopen("MNIST/t10k-labels.idx1-ubyte", "r");
  if (y_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(y_fp, 8, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  // XXX hacky as hell
  const int total = 10000;
  int correct = 0;

  // XXX code dup
  for (int batch = 0; batch < total; batch++) {
    ARRAY_FOR(x) elem = (double)fgetc(x_fp) / 256;
    ARRAY_FOR(y) elem = 0.0;
    int yi = fgetc(y_fp); // XXX bounds check
    y[yi] = 1.0;

    predict(x, w, yh);
    int yhi = 0;
    ARRAY_FOR(yh) if (elem > yh[yhi]) yhi = idx;
    correct += yhi == yi;

    if (yi == yhi)
      continue;

    printf("yh = ");
    ARRAY_FOR(yh) printf("%f ", elem);
    putchar('\n');
    printf("yhi = %d\n", yhi);
    printf("yi = %d\n", yi);
    ARRAY_FOR(x) {
      char ch = " .,+*o%@"[(int)(elem * 8)];
      putchar(ch), putchar(ch);
      (idx + 1) % 28 || putchar('\n');
    }
  }

  printf("MNIST accuracy: %f\n", (double)correct / total);

  if (fclose(x_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  if (fclose(y_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  return 0;
}
