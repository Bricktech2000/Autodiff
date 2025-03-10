#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// XXX code dup
void predict(double x[], double w[], double y[]);
void backprop(double x[], double w[], double yh[], double dw[], double *c);

// // faster
// #define ALPHA 0.5
// #define BATCH 100
// #define EPOCHS 10000

// slower
#define ALPHA 0.1
#define BATCH 250
#define EPOCHS 10000

// // broken
// #define ALPHA 0.5
// #define BATCH 1000
// #define EPOCHS 1000

int main(void) {
  srand(time(NULL));

  FILE *x_fp = fopen("MNIST/train-images.idx3-ubyte", "r");
  if (x_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(x_fp, 16, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  FILE *yh_fp = fopen("MNIST/train-labels.idx1-ubyte", "r");
  if (yh_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(yh_fp, 8, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  static double x[784];
  static double w[16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10];
  static double yh[10];
  static double dw[16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10];
  static double c;

  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    w[i] = (double)rand() / RAND_MAX * 2 - 1;

  for (int e = 0; e < EPOCHS; e++) {
    for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
      dw[i] = 0.0;
    c = 0.0;

    for (int b = 0; b < BATCH; b++) {
      for (int i = 0; i < 784; i++)
        x[i] = (double)fgetc(x_fp) / 256;
      for (int i = 0; i < 10; i++)
        yh[i] = 0.0;
      int yhi = fgetc(yh_fp); // XXX bounds check
      yh[yhi] = 1.0;

      backprop(x, w, yh, dw, &c);
    }

    // XXX hacky as hell
    if (ftell(yh_fp) >= 50000)
      fseek(x_fp, 16, SEEK_SET), fseek(yh_fp, 8, SEEK_SET);

    for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
      w[i] -= ALPHA * dw[i];
    c /= BATCH;

    printf("epoch %d of %d; loss %f\n", e, EPOCHS, c);
  }

  if (fclose(x_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  if (fclose(yh_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  x_fp = fopen("MNIST/t10k-images.idx3-ubyte", "r");
  if (x_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(x_fp, 16, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  yh_fp = fopen("MNIST/t10k-labels.idx1-ubyte", "r");
  if (yh_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(yh_fp, 8, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  // XXX hacky as hell
  const int n = 10000;
  int good = 0;

  // XXX code dup
  for (int b = 0; b < n; b++) {
    for (int i = 0; i < 784; i++)
      x[i] = (double)fgetc(x_fp) / 256;
    for (int i = 0; i < 10; i++)
      yh[i] = 0.0;
    int yhi = fgetc(yh_fp); // XXX bounds check
    yh[yhi] = 1.0;

    // printout
    printf("x =\n");
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        char ch = " .,+*o%@"[(int)(x[i * 28 + j] * 8)];
        putchar(ch), putchar(ch);
      }
      putchar('\n');
    }
    double y[10];
    predict(x, w, y);
    printf("y =\n");
    for (int i = 0; i < 10; i++)
      printf("%f ", y[i]);
    putchar('\n');

    // accuracy
    int yi = 0.0;
    double max = -INFINITY;
    for (int i = 0; i < 10; i++)
      if (y[i] > max)
        max = y[i], yi = i;
    printf("yi = %d\n", yi);
    printf("yhi = %d\n", yhi);
    good += yi == yhi;
  }

  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    printf("%f ", dw[i]);
  printf("\n");

  printf("MNIST accuracy: %f\n", (double)good / n);

  if (fclose(x_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  if (fclose(yh_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  return 0;
}
