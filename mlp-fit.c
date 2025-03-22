#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <time.h>

#define THREADS 16

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

struct arg {
  unsigned seed;
  struct ex *exs;
  w_t *w;
  dw_t *dw;
  c_t *c;
};

mtx_t sync_lock, grad_lock;
cnd_t work_avail, work_done;
int work_remain = THREADS;

// ISO/IEC 9899:TC3, $7.20.2.2, paragraph 5
#define RAND_R_MAX 32767
int rand_r(unsigned *seedp) {
  *seedp = *seedp * 1103515245 + 12345;
  return *seedp / 65536 % 32768;
}

int worker(void *arg) {
  struct arg *a = arg;
  unsigned seed = a->seed;
  dw_t dw;
  c_t c;

  while (1) {
    mtx_lock(&sync_lock);
    work_remain--;
    cnd_signal(&work_done);
    cnd_wait(&work_avail, &sync_lock);
    if (work_remain == -1) // terminate request
      break;
    mtx_unlock(&sync_lock);

    ARRAY_FOR(dw) elem = 0.0;
    *c = 0.0;

    for (int batch = 0; batch < BATCH / THREADS; batch++) {
      // standard `rand()` is not required to be thread safe
      size_t rand = (size_t)rand_r(&seed) * (RAND_R_MAX + 1) + rand_r(&seed);
      struct ex *ex = a->exs + rand % TRAIN_LEN;
      mlp_backprop(ex->x, *a->w, ex->y, dw, c);
    }

    ARRAY_FOR(dw) elem /= BATCH;
    *c /= BATCH;

    mtx_lock(&grad_lock);
    ARRAY_FOR(*a->dw) elem += dw[idx];
    **a->c += *c;
    mtx_unlock(&grad_lock);
  }

  mtx_unlock(&sync_lock);
  return 0;
}

int main(void) {
  srand(time(NULL));

  struct ex *train_exs = load_mnist(TRAIN_PATHS, TRAIN_LEN);
  struct ex *test_exs = load_mnist(TEST_PATHS, TEST_LEN);

  static w_t w;
  static yh_t yh;
  static dw_t dw;
  static c_t c;
  static dw_t v;

  ARRAY_FOR(v) elem = 0.0;
  ARRAY_FOR(w) elem = (double)rand() / RAND_MAX - 0.5;

  mtx_init(&sync_lock, mtx_plain), mtx_init(&grad_lock, mtx_plain);
  cnd_init(&work_avail), cnd_init(&work_done);

  struct arg args[THREADS];
  thrd_t thrds[THREADS];
  for (int i = 0; i < THREADS; i++) {
    args[i] = (struct arg){rand(), train_exs, &w, &dw, &c};
    thrd_create(thrds + i, worker, args + i);
  }

  // make sure all threads are waiting
  mtx_lock(&sync_lock);
  while (work_remain)
    cnd_wait(&work_done, &sync_lock);
  mtx_unlock(&sync_lock);

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    ARRAY_FOR(dw) elem = 0.0;
    *c = 0.0;

    mtx_lock(&sync_lock);
    work_remain = THREADS;
    cnd_broadcast(&work_avail);
    while (work_remain)
      cnd_wait(&work_done, &sync_lock);
    mtx_unlock(&sync_lock);

    mtx_lock(&grad_lock);
    ARRAY_FOR(dw) elem += LAMBDA * w[idx] * w[idx];  // L2 regularization
    ARRAY_FOR(v) elem = elem * BETA - ETA * dw[idx]; // momentum
    ARRAY_FOR(w) elem += v[idx];                     // gradient descent

    printf("epoch %d of %d; loss %f", epoch, EPOCHS, *c);
    printf("%*s\n", (int)(*c * 64), "#");
    mtx_unlock(&grad_lock);
  }

  mtx_lock(&sync_lock);
  work_remain = -1; // terminate request
  cnd_broadcast(&work_avail);
  mtx_unlock(&sync_lock);

  for (int i = 0; i < THREADS; i++)
    thrd_join(thrds[i], NULL);

  mtx_destroy(&sync_lock), mtx_destroy(&grad_lock);
  cnd_destroy(&work_avail), cnd_destroy(&work_done);

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
