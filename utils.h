#ifndef UTILS_H
#define UTILS_H

void matmul(const float* A, const float* B, float* C, int M, int N, int K);
void softmax(float* x, int len);
float randf();

#endif 