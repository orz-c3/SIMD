#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h> //AVX and AVX2
#include <windows.h>
//#include <sys/time.h>
using namespace std;

const int n = 2000;
float** A;

// 分配对齐内存并初始化矩阵

void init_align() {
    // 分配对齐内存
    A = (float**)_aligned_malloc(n * sizeof(float*), 32); // 32-byte 对齐，用于 AVX
    for (int i = 0; i < n; ++i) {
        A[i] = (float*)_aligned_malloc(n * sizeof(float), 32); // 32-byte 对齐，用于 AVX
    }

    // 初始化矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = 0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            A[i][j] = rand() % 1000;
        }
    }

    // 对矩阵进行累加操作，确保数据有一定的复杂度
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}

void init_non_align() {
    // 分配非对齐内存
    A = (float**)malloc(n * sizeof(float*));  // 使用标准的malloc分配指针数组
    for (int i = 0; i < n; ++i) {
        A[i] = (float*)malloc(n * sizeof(float)); // 使用标准的malloc为每一行分配内存
    }

    // 初始化矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = 0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            A[i][j] = rand() % 1000;
        }
    }

    // 对矩阵进行累加操作，确保数据有一定的复杂度
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}

// 释放分配的内存
void cleanup() {
    for (int i = 0; i < n; ++i) {
        _aligned_free(A[i]);
    }
    _aligned_free(A);
}

// 普通版本的矩阵运算
void f_ordinary() {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j) {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

// 使用 SSE 指令集的矩阵运算
void f_sse() {
    for (int k = 0; k < n; ++k) {
        __m128 vt = _mm_set1_ps(A[k][k]);
        int j;
        for (j = k + 1; j + 4 <= n; j += 4) {
            __m128 va = _mm_loadu_ps(&(A[k][j]));
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&(A[k][j]), va);
        }

        for (; j < n; ++j) {
            A[k][j] = A[k][j] * 1.0f / A[k][k];
        }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            __m128 vaik = _mm_set1_ps(A[i][k]);

            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 vakj = _mm_loadu_ps(&(A[k][j]));
                __m128 vaij = _mm_loadu_ps(&(A[i][j]));
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i][j], vaij);
            }

            for (; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

            A[i][k] = 0.0f;
        }
    }
}

void f_sse_align() {
    for (int k = 0; k < n; ++k) {
        __m128 vt = _mm_set1_ps(A[k][k]);  // Set vt to contain multiple copies of A[k][k]
        int j = k + 1;

        // Scalar handling to ensure alignment for SSE operations
        while ((uintptr_t)(&A[k][j]) % 16 != 0 && j < n) {
            A[k][j] = A[k][j] * 1.0f / A[k][k];  // Handle elements until we reach alignment
            j++;
        }

        // Now j is aligned, perform vectorized operations
        for (; j + 4 <= n; j += 4) {
            __m128 va = _mm_load_ps(&A[k][j]);  // Load data with _mm_load_ps assuming data is aligned
            va = _mm_div_ps(va, vt);
            _mm_store_ps(&A[k][j], va);  // Store data with _mm_store_ps assuming data is aligned
        }

        // Handle any remaining elements after the last full vector
        for (; j < n; ++j) {
            A[k][j] = A[k][j] * 1.0f / A[k][k];
        }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            __m128 vaik = _mm_set1_ps(A[i][k]);

            // Repeat alignment check for each row
            j = k + 1;
            while ((uintptr_t)(&A[i][j]) % 16 != 0 && j < n) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];  // Scalar operation until alignment
                j++;
            }

            for (; j + 4 <= n; j += 4) {
                __m128 vakj = _mm_load_ps(&A[k][j]);  // Assumed data is aligned
                __m128 vaij = _mm_load_ps(&A[i][j]);  // Assumed data is aligned
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_store_ps(&A[i][j], vaij);  // Store data assuming it is aligned
            }

            // Handle any remaining elements
            for (; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

            A[i][k] = 0.0f;
        }
    }
}

// 使用 AVX 指令集的矩阵运算
void f_avx256() {
    for (int k = 0; k < n; ++k) {
        __m256 vt2 = _mm256_set1_ps(A[k][k]);
        int j;
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 va2 = _mm256_loadu_ps(&(A[k][j]));
            va2 = _mm256_div_ps(va2, vt2);
            _mm256_storeu_ps(&(A[k][j]), va2);
        }

        for (; j < n; ++j) {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; ++i) {
            __m256 vaik2 = _mm256_set1_ps(A[i][k]);

            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vakj2 = _mm256_loadu_ps(&(A[k][j]));
                __m256 vaij2 = _mm256_loadu_ps(&(A[i][j]));
                __m256 vx2 = _mm256_mul_ps(vakj2, vaik2);
                vaij2 = _mm256_sub_ps(vaij2, vx2);
                _mm256_storeu_ps(&A[i][j], vaij2);
            }

            for (; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

            A[i][k] = 0;
        }
    }
}

void f_avx256_align() {
    for (int k = 0; k < n; ++k) {
        __m256 vt = _mm256_set1_ps(A[k][k]);  // Set vt to contain multiple copies of A[k][k]
        int j = k + 1;

        // Scalar handling to ensure alignment for SSE operations
        while ((uintptr_t)(&A[k][j]) % 32 != 0 && j < n) {
            A[k][j] = A[k][j] * 1.0f / A[k][k];  // Handle elements until we reach alignment
            j++;
        }

        // Now j is aligned, perform vectorized operations
        for (; j + 8 <= n; j += 8) {
            __m256 va = _mm256_load_ps(&A[k][j]);  // Load data with _mm_load_ps assuming data is aligned
            va = _mm256_div_ps(va, vt);
            _mm256_store_ps(&A[k][j], va);  // Store data with _mm_store_ps assuming data is aligned
        }

        // Handle any remaining elements after the last full vector
        for (; j < n; ++j) {
            A[k][j] = A[k][j] * 1.0f / A[k][k];
        }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            __m256 vaik = _mm256_set1_ps(A[i][k]);

            // Repeat alignment check for each row
            j = k + 1;
            while ((uintptr_t)(&A[i][j]) % 32 != 0 && j < n) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];  // Scalar operation until alignment
                j++;
            }

            for (; j + 8 <= n; j += 8) {
                __m256 vakj = _mm256_load_ps(&A[k][j]);  // Assumed data is aligned
                __m256 vaij = _mm256_load_ps(&A[i][j]);  // Assumed data is aligned
                __m256 vx = _mm256_mul_ps(vaik, vakj);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_store_ps(&A[i][j], vaij);  // Store data assuming it is aligned
            }

            // Handle any remaining elements
            for (; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

            A[i][k] = 0.0f;
        }
    }
}

// 使用 AVX-512 指令集的矩阵运算
void f_avx512() {
    for (int k = 0; k < n; ++k) {
        __m512 vt3 = _mm512_set1_ps(A[k][k]);
        int j;
        for (j = k + 1; j + 16 <= n; j += 16) {
            __m512 va3 = _mm512_loadu_ps(&(A[k][j]));
            va3 = _mm512_div_ps(va3, vt3);
            _mm512_store_ps(&(A[k][j]), va3);
        }

        for (; j < n; ++j) {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; ++i) {
            __m512 vaik3 = _mm512_set1_ps(A[i][k]);

            for (j = k + 1; j + 16 <= n; j += 16) {
                __m512 vakj3 = _mm512_loadu_ps(&(A[k][j]));
                __m512 vaij3 = _mm512_loadu_ps(&(A[i][j]));
                __m512 vx3 = _mm512_mul_ps(vakj3, vaik3);
                vaij3 = _mm512_sub_ps(vaij3, vx3);
                _mm512_store_ps(&A[i][j], vaij3);
            }

            for (; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

            A[i][k] = 0;
        }
    }
}

// 主函数
int main() {
    // 测试每个函数的性能
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    init_non_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_ordinary();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_ordinary:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;

    init_non_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_sse();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_sse:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;
    
    init_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_sse_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_sse_align:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;

    init_non_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_avx256();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_avx256:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;

    init_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_avx256_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_avx256_align:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;

    init_align();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f_avx512();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_avx512:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;

    cleanup(); // 释放分配的内存
    return 0;
}
