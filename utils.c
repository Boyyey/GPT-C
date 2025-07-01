#include "utils.h"
#include <stdlib.h>
#include <math.h>

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void softmax(float* x, int len) {
    float max = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}

float randf() {
    return (float)rand() / (float)RAND_MAX;
} 