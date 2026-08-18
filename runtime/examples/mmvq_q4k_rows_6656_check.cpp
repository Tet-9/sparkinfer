#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/quant.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}
template <class T> T* device_copy(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> bool equal_device(const T* a, const T* b, size_t n, const char* what) {
    std::vector<T> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (ha == hb) return true;
    size_t i = 0;
    while (i < n && ha[i] == hb[i]) i++;
    std::printf("[FAIL] %s differs at %zu (%d vs %d)\n", what, i, (int)ha[i], (int)hb[i]);
    return false;
}
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("[SKIP] no CUDA device\n");
        return 0;
    }
    constexpr int N = 4, MH = 6656, MN = 8192, SB = MH / 256; // SB=26, Muse Glimmer's hidden size
    auto make_bf16 = [](size_t n, int salt, float scale) {
        std::vector<uint16_t> h(n);
        for (size_t i = 0; i < n; i++) {
            const int v = (int)((i * 1103515245u + 12345u + salt) % 2001u) - 1000;
            h[i] = bf16(scale * v / 1000.f);
        }
        return h;
    };
    auto hact = make_bf16((size_t)N * MH, 11, 0.2f);
    uint16_t* dact = device_copy(hact);
    const size_t q81_row = sparkinfer::kernels::llama_q8_1_bytes(MH);
    void* dq81 = nullptr;
    cudaMalloc(&dq81, (size_t)N * q81_row);
    sparkinfer::kernels::launch_quantize_q8_1_rows(dact, dq81, MH, N, MH);

    std::vector<unsigned char> hw((size_t)MN * SB * 144);
    for (int row = 0; row < MN; row++) for (int sb = 0; sb < SB; sb++) {
        unsigned char* p = hw.data() + ((size_t)row * SB + sb) * 144;
        p[0] = 0x1f; p[1] = 0x21; p[2] = 0x1f; p[3] = 0x21;
        for (int i = 4; i < 144; i++) p[i] = (unsigned char)((row * 17 + sb * 29 + i * 13) & 255);
    }
    unsigned char* dw = device_copy(hw);
    uint16_t *ym = nullptr, *ys = nullptr;
    cudaMalloc(&ym, (size_t)N * MN * 2); cudaMalloc(&ys, (size_t)N * MN * 2);

    bool ok = sparkinfer::kernels::launch_mmvq_q4k_rows(dq81, dw, ym, N, MN, MH);
    if (!ok) { std::printf("[FAIL] launch_mmvq_q4k_rows returned false at K=%d\n", MH); return 1; }
    for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q4k((const char*)dq81 + (size_t)m * q81_row,
                                              dw, ys + (size_t)m * MN, MN, MH);
    cudaDeviceSynchronize();
    ok = equal_device(ym, ys, (size_t)N * MN, "K=6656 Q4_K exact rows") && ok;

    // Timing: batched rows vs. serial single-row calls
    cudaEvent_t t0, t1, t2;
    cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2);
    cudaEventRecord(t0);
    for (int rep = 0; rep < 200; rep++)
        sparkinfer::kernels::launch_mmvq_q4k_rows(dq81, dw, ym, N, MN, MH);
    cudaEventRecord(t1);
    for (int rep = 0; rep < 200; rep++)
        for (int m = 0; m < N; m++)
            sparkinfer::kernels::launch_mmvq_q4k((const char*)dq81 + (size_t)m * q81_row,
                                                  dw, ys + (size_t)m * MN, MN, MH);
    cudaEventRecord(t2);
    cudaEventSynchronize(t2);
    float ms_batch = 0, ms_serial = 0;
    cudaEventElapsedTime(&ms_batch, t0, t1);
    cudaEventElapsedTime(&ms_serial, t1, t2);
    std::printf("K=6656 rows: batch %.4f ms, %d serial calls %.4f ms, %.2fx\n",
                ms_batch / 200, N, ms_serial / 200, ms_serial / ms_batch);

    std::printf(ok ? "[PASS] K=6656 batched-row Q4_K matches serial reference\n"
                    : "[FAIL] K=6656 mismatch\n");
    return ok ? 0 : 1;
}
