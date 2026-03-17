#include <complex.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>

#include <lean/lean.h>

extern void zgemm_(const char *transa, const char *transb,
                   const int *m, const int *n, const int *k,
                   const double _Complex *alpha,
                   const double _Complex *a, const int *lda,
                   const double _Complex *b, const int *ldb,
                   const double _Complex *beta,
                   double _Complex *c, const int *ldc);
extern void zsyrk_(const char *uplo, const char *trans,
                   const int *n, const int *k,
                   const double _Complex *alpha,
                   const double _Complex *a, const int *lda,
                   const double _Complex *beta,
                   double _Complex *c, const int *ldc);
extern void zgesv_(const int *n, const int *nrhs,
                   double _Complex *a, const int *lda, int *ipiv,
                   double _Complex *b, const int *ldb, int *info);

static inline lean_obj_res mk_sized_float_array(size_t size) {
    return lean_alloc_sarray(sizeof(double), size, size);
}

static inline lean_obj_res mk_empty_float_array(void) {
    return mk_sized_float_array(0);
}

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

LEAN_EXPORT lean_obj_res lean_covariant_complex_gram(
    size_t rows, size_t cols, size_t mode, b_lean_obj_arg entries
) {
    size_t entry_count = rows * cols;
    if (rows > (size_t)INT_MAX || cols > (size_t)INT_MAX) {
        return mk_empty_float_array();
    }
    if (lean_sarray_size(entries) != 2 * entry_count) {
        return mk_empty_float_array();
    }
    if (cols == 0) {
        return mk_empty_float_array();
    }

    const double *src = lean_float_array_cptr(entries);
    double _Complex *a = copy_complex_entries(entry_count, src);
    double _Complex *c = (double _Complex *)malloc(cols * cols * sizeof(double _Complex));
    if (a == NULL || c == NULL) {
        free(a);
        free(c);
        return mk_empty_float_array();
    }

    int m = (int)rows;
    int n = (int)cols;
    int lda = m;
    int ldc = n;
    double _Complex alpha = 1.0 + 0.0 * I;
    double _Complex beta = 0.0 + 0.0 * I;

    if (mode == 0) {
        char transa = 'C';
        char transb = 'N';
        zgemm_(&transa, &transb, &n, &n, &m, &alpha, a, &lda, a, &lda, &beta, c, &ldc);
    } else if (mode == 1) {
        char uplo = 'U';
        char trans = 'T';
        zsyrk_(&uplo, &trans, &n, &m, &alpha, a, &lda, &beta, c, &ldc);
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < col; ++row) {
                c[col + (size_t)row * (size_t)n] = c[row + (size_t)col * (size_t)n];
            }
        }
    } else {
        free(a);
        free(c);
        return mk_empty_float_array();
    }

    lean_obj_res out = mk_sized_float_array(2 * cols * cols);
    double *dst = lean_float_array_cptr(out);
    for (size_t i = 0; i < cols * cols; ++i) {
        dst[2 * i] = creal(c[i]);
        dst[2 * i + 1] = cimag(c[i]);
    }

    free(a);
    free(c);
    return out;
}

LEAN_EXPORT lean_obj_res lean_covariant_complex_solve(
    size_t n, b_lean_obj_arg entries, b_lean_obj_arg rhs_entries
) {
    if (n > (size_t)INT_MAX) {
        return mk_empty_float_array();
    }
    if (lean_sarray_size(entries) != 2 * n * n || lean_sarray_size(rhs_entries) != 2 * n) {
        return mk_empty_float_array();
    }
    if (n == 0) {
        return mk_empty_float_array();
    }

    const double *a_src = lean_float_array_cptr(entries);
    const double *rhs_src = lean_float_array_cptr(rhs_entries);
    double _Complex *a = copy_complex_entries(n * n, a_src);
    double _Complex *rhs = copy_complex_entries(n, rhs_src);
    int *ipiv = (int *)malloc(n * sizeof(int));
    if (a == NULL || rhs == NULL || ipiv == NULL) {
        free(a);
        free(rhs);
        free(ipiv);
        return mk_empty_float_array();
    }

    int dim = (int)n;
    int nrhs = 1;
    int info = 0;
    zgesv_(&dim, &nrhs, a, &dim, ipiv, rhs, &dim, &info);

    free(a);
    free(ipiv);
    if (info != 0) {
        free(rhs);
        return mk_empty_float_array();
    }

    lean_obj_res out = mk_sized_float_array(2 * n);
    double *dst = lean_float_array_cptr(out);
    for (size_t i = 0; i < n; ++i) {
        dst[2 * i] = creal(rhs[i]);
        dst[2 * i + 1] = cimag(rhs[i]);
    }
    free(rhs);
    return out;
}
