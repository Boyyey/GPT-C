#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

Tensor* tensor_create(int* shape, int ndim);
void tensor_free(Tensor* t);
void tensor_print(Tensor* t);

#endif 