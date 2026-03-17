#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include <lean/lean.h>

extern void zgetrf_(const int *m, const int *n, double _Complex *a, const int *lda, int *ipiv, int *info);
extern void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);

static double _Complex *copy_complex_entries(size_t count, const double *src) {
    double _Complex *dst = (double _Complex *)malloc(count * sizeof(double _Complex));
    if (dst == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[2 * i] + src[2 * i + 1] * I;
    }
    return dst;
}

static double log_abs_det_lu(size_t n, double _Complex *a, int *ipiv) {
    int dim = (int)n;
    int info = 0;
    zgetrf_(&dim, &dim, a, &dim, ipiv, &info);
    if (info != 0) {
        return NAN;
    }

    double log_abs_det = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double mag = cabs(a[i + i * n]);
        if (!(mag > 0.0) || !isfinite(mag)) {
            return NAN;
        }
        log_abs_det += log(mag);
    }
    return log_abs_det;
}

LEAN_EXPORT double lean_covariant_complex_log_abs_det(size_t n, b_lean_obj_arg entries) {
    if (n > (size_t)INT_MAX) {
        return NAN;
    }
    if (lean_sarray_size(entries) != 2 * n * n) {
        return NAN;
    }
    if (n == 0) {
        return 0.0;
    }

    const double *src = lean_float_array_cptr(entries);
    double _Complex *a = copy_complex_entries(n * n, src);
    int *ipiv = (int *)malloc(n * sizeof(int));
    if (a == NULL || ipiv == NULL) {
        free(a);
        free(ipiv);
        return NAN;
    }

    double log_abs_det = log_abs_det_lu(n, a, ipiv);
    free(a);
    free(ipiv);
    return log_abs_det;
}

LEAN_EXPORT double lean_covariant_real_spd_log_det(size_t n, b_lean_obj_arg entries) {
    if (n > (size_t)INT_MAX) {
        return NAN;
    }
    if (lean_sarray_size(entries) != n * n) {
        return NAN;
    }
    if (n == 0) {
        return 0.0;
    }

    const double *src = lean_float_array_cptr(entries);
    double *a = (double *)malloc(n * n * sizeof(double));
    if (a == NULL) {
        return NAN;
    }
    for (size_t i = 0; i < n * n; ++i) {
        a[i] = src[i];
    }

    char uplo = 'L';
    int dim = (int)n;
    int info = 0;
    dpotrf_(&uplo, &dim, a, &dim, &info);
    if (info != 0) {
        free(a);
        return NAN;
    }

    double logdet = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diag = a[i + i * n];
        if (!(diag > 0.0) || !isfinite(diag)) {
            free(a);
            return NAN;
        }
        logdet += 2.0 * log(diag);
    }

    free(a);
    return logdet;
}
