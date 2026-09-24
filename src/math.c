#include <math.h>
#include "ops.h"

void matmul(const float *a, const float *b, float *out, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < k; p++) {
                s += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
}

void add(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void layernorm(const float *x, const float *gamma, const float *beta, float *out, int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;

    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }
}

void softmax(const float *x, float *out, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

void gelu(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
    }
}

void linear(const float *x, const float *w, const float *b, float *out, int in_dim, int out_dim) {
    matmul(x, w, out, 1, in_dim, out_dim);
    add(out, b, out, out_dim);
}
