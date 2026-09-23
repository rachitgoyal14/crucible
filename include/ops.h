#ifndef MATH_H
#define MATH_H

void matmul(const float *a, const float *b, float *out, int m, int k, int n);
void add(const float *a, const float *b, float *out, int n);
void layernorm(const float *x, const float *gamma, const float *beta, float *out, int n, float eps);
void softmax(const float *x, float *out, int n);
void gelu(const float *x, float *out, int n);
void linear(const float *x, const float *w, const float *b, float *out, int in_dim, int out_dim);

#endif
