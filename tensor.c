#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

Tensor* tensor_create(int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    t->size = 1;
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    t->data = (float*)calloc(t->size, sizeof(float));
    return t;
}

void tensor_free(Tensor* t) {
    free(t->data);
    free(t->shape);
    free(t);
}

void tensor_print(Tensor* t) {
    for (int i = 0; i < t->size; ++i) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
} 