#include <math.h>
#include <stdio.h>
#include "ops.h"

static int fails = 0;

static void expect(const float *got, const float *want, int n, float tol, const char *name) {
    for (int i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) {
            printf("FAIL %s[%d]: got %f want %f\n", name, i, got[i], want[i]);
            fails++;
            return;
        }
    }
    printf("ok %s\n", name);
}

int main(void) {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {7, 8, 9, 10, 11, 12};
    float c[4] = {0};
    float want_mm[4] = {58, 64, 139, 154};
    matmul(a, b, c, 2, 3, 2);
    expect(c, want_mm, 4, 1e-5f, "matmul 2x3x2");

    float x[3] = {1.5f, -2.0f, 3.0f};
    float y[3] = {0.5f, 2.0f, 1.0f};
    float z[3] = {0};
    float want_add[3] = {2.0f, 0.0f, 4.0f};
    add(x, y, z, 3);
    expect(z, want_add, 3, 1e-6f, "add");

    float v[4] = {1, 2, 3, 4};
    float g[4] = {1, 1, 1, 1};
    float bi[4] = {0, 0, 0, 0};
    float o[4] = {0};
    float want_ln[4] = {-1.3416407f, -0.4472136f, 0.4472136f, 1.3416407f};
    layernorm(v, g, bi, o, 4, 1e-5f);
    expect(o, want_ln, 4, 1e-4f, "layernorm");

    float s[3] = {1000.0f, 1001.0f, 999.0f};
    float p[3] = {0};
    float want_sm[3] = {0.244728f, 0.665241f, 0.090031f};
    softmax(s, p, 3);
    expect(p, want_sm, 3, 1e-4f, "softmax stable");

    float gl[3] = {0.0f, 1.0f, -1.0f};
    float go[3] = {0};
    float want_gelu[3] = {0.0f, 0.841192f, -0.158808f};
    gelu(gl, go, 3);
    expect(go, want_gelu, 3, 1e-4f, "gelu tanh");

    float lx[3] = {1, 2, 3};
    float lw[6] = {7, 8, 9, 10, 11, 12};
    float lb[2] = {1, 1};
    float lo[2] = {0};
    float want_lin[2] = {59, 65};
    linear(lx, lw, lb, lo, 3, 2);
    expect(lo, want_lin, 2, 1e-5f, "linear");

    if (fails) {
        printf("%d failures\n", fails);
        return 1;
    }
    printf("all pass\n");
    return 0;
}
