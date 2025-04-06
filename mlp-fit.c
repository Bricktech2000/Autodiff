#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef __STDC_NO_THREADS__
#include <threads.h>
#ifndef __STDC_NO_ATOMICS__
#include <stdatomic.h>
#endif
#endif

#define THREADS 16

// // faster (90% accuracy)
// #define ETA 0.05   // learning rate
// #define BETA 0.9   // momentum coefficient
// #define LAMBDA 0.0 // regularization rate
// #define BATCH 100  // mini-batch size
// #define ITERS 500  // number of gradient updates

// slower (96% accuracy)
#define ETA 0.01    // learning rate
#define BETA 0.9    // momentum coefficient
#define LAMBDA 0.0  // regularization rate
#define BATCH 250   // mini-batch size
#define ITERS 10000 // number of update steps

#define TRAIN_LEN 60000
#define TRAIN_OFSTS 16, 8
#define TRAIN_PATHS                                                            \
  "MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte"
#define TEST_LEN 10000
#define TEST_OFSTS 16, 8
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

struct ex *load_mnist(char *x_path, char *y_path, long x_ofst, long y_ofst,
                      size_t len) {
  FILE *x_fp = fopen(x_path, "r"), *y_fp = fopen(y_path, "r");
  if (x_fp == NULL || y_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  if (fseek(x_fp, x_ofst, SEEK_SET) == EOF ||
      fseek(y_fp, y_ofst, SEEK_SET) == EOF)
    perror("fseek"), exit(EXIT_FAILURE);

  struct ex *exs = malloc(sizeof(*exs) * len);

  int chr;
  for (size_t i = 0; i < len; i++) {
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

  if (fclose(x_fp) == EOF || fclose(y_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  return exs;
}

int mnist_y_to_yi(y_t *y) {
  int yi = 0;
  ARRAY_FOR(*y) yi = elem > (*y)[yi] ? idx : yi;
  return yi;
}

void mnist_x_dump(x_t *x) {
  ARRAY_FOR(*x) {
    char tone = " .,+*o%@"[(int)(elem * 8.0)];
    printf("%c%c", tone, tone);
    (idx + 1) % 28 || putchar('\n');
  }
}

void mnist_y_dump(y_t *y) {
  ARRAY_FOR(*y) printf("%f ", elem);
  putchar('\n');
}

#ifndef __STDC_NO_THREADS__
struct arg {
  unsigned seed;
  struct ex *exs;
  w_t *w;
  dw_t *dw;
  c_t *c;
};

mtx_t sync_lock;
cnd_t work_avail, work_done;
int thrds_working;
#ifndef __STDC_NO_ATOMICS__
_Atomic int exs_left;
#endif

// ISO/IEC 9899:TC3, $7.20.2.2, paragraph 5
#define RAND_R_MAX 32767
int rand_r(unsigned *seedp) {
  *seedp = *seedp * 1103515245 + 12345;
  return *seedp / 65536 % 32768;
}

int worker_thrd(void *arg) {
  struct arg *a = arg;
  unsigned seed = a->seed;
  dw_t dw;
  c_t c;

  mtx_lock(&sync_lock);

  while (thrds_working != EOF) {
    mtx_unlock(&sync_lock);

    ARRAY_FOR(dw) elem = 0.0;
    *c = 0.0;

#ifndef __STDC_NO_ATOMICS__
    while (atomic_fetch_sub_explicit(&exs_left, 1, memory_order_relaxed) > 0)
#else
    for (int ex = 0; ex < BATCH / THREADS; ex++)
#endif
    {
      // standard `rand()` is not required to be thread safe
      size_t idx = rand_r(&seed) * (RAND_R_MAX + (size_t)1) + rand_r(&seed);
      struct ex *ex = a->exs + idx % TRAIN_LEN;
      mlp_backprop(ex->x, *a->w, ex->y, dw, c);
    }

    ARRAY_FOR(dw) elem /= BATCH;
    *c /= BATCH;

    mtx_lock(&sync_lock);

    ARRAY_FOR(*a->dw) elem += dw[idx];
    **a->c += *c;

    thrds_working--;
    cnd_signal(&work_done);
    cnd_wait(&work_avail, &sync_lock);
  }

  mtx_unlock(&sync_lock);

  return 0;
}
#endif // __STDC_NO_THREADS__

int main(void) {
  srand(time(NULL));

  struct ex *train_exs = load_mnist(TRAIN_PATHS, TRAIN_OFSTS, TRAIN_LEN);
  struct ex *test_exs = load_mnist(TEST_PATHS, TEST_OFSTS, TEST_LEN);

  static w_t w;
  static yh_t yh;
  static dw_t dw;
  static c_t c;
  static dw_t v;

  ARRAY_FOR(v) elem = 0.0;
  ARRAY_FOR(w) elem = (double)rand() / RAND_MAX - 0.5;

#ifndef __STDC_NO_THREADS__
  mtx_init(&sync_lock, mtx_plain);
  cnd_init(&work_avail), cnd_init(&work_done);

  mtx_lock(&sync_lock);

  struct arg args[THREADS];
  thrd_t thrds[THREADS];
  for (int i = 0; i < THREADS; i++) {
    args[i] = (struct arg){rand(), train_exs, &w, &dw, &c};
    thrd_create(thrds + i, worker_thrd, args + i);
  }
#endif

  for (int iter = 0; iter < ITERS; iter++) {
    ARRAY_FOR(dw) elem = 0.0;
    *c = 0.0;

#ifndef __STDC_NO_THREADS__
#ifndef __STDC_NO_ATOMICS__
    exs_left = BATCH;
#endif
    thrds_working = THREADS;
    cnd_broadcast(&work_avail);
    while (thrds_working)
      cnd_wait(&work_done, &sync_lock);
#else
    for (int ex = 0; ex < BATCH; ex++) {
      // ISO/IEC 9899:TC3, $7.20.2.1 requires RAND_MAX to be at least 32767
      size_t idx = rand() * (RAND_MAX + (size_t)1) + rand();
      struct ex *ex = train_exs + idx % TRAIN_LEN;
      mlp_backprop(ex->x, w, ex->y, dw, c);
    }

    ARRAY_FOR(dw) elem /= BATCH;
    *c /= BATCH;
#endif

    ARRAY_FOR(dw) elem += LAMBDA * w[idx] * w[idx];  // L2 regularization
    ARRAY_FOR(v) elem = elem * BETA - ETA * dw[idx]; // momentum
    ARRAY_FOR(w) elem += v[idx];                     // gradient descent

    printf("iter %d of %d; loss %f", iter, ITERS, *c);
    printf("%*s\n", (int)(*c * 64), "#");
  }

#ifndef __STDC_NO_THREADS__
  thrds_working = EOF;
  cnd_broadcast(&work_avail);
  mtx_unlock(&sync_lock);

  for (int i = 0; i < THREADS; i++)
    thrd_join(thrds[i], NULL);

  mtx_destroy(&sync_lock);
  cnd_destroy(&work_avail), cnd_destroy(&work_done);
#endif

  double accuracy = 0.0;

  for (size_t i = 0; i < TEST_LEN; i++) {
    struct ex *ex = test_exs + i;
    mlp_predict(ex->x, w, yh);

    int correct = mnist_y_to_yi(&yh) == mnist_y_to_yi(&ex->y);
    accuracy += (double)correct / TEST_LEN;

    if (correct)
      continue;

    printf("yh  = "), mnist_y_dump(&yh);
    printf("y   = "), mnist_y_dump(&ex->y);
    printf("yhi = %d\n", mnist_y_to_yi(&yh));
    printf("yi  = %d\n", mnist_y_to_yi(&ex->y));
    mnist_x_dump(&ex->x);
  }

  printf("accuracy: %f\n", accuracy);

  free(train_exs), free(test_exs);
}
