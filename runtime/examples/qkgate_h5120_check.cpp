#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/fused.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
uint16_t bf16_bits(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}
template <class T> T* device_copy(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("[SKIP] no CUDA device\n");
        return 0;
    }
    // Qwen3.8-27B Gated Attention shape (per published config): 24 Q heads, 4 KV heads,
    // head_dim=256, rotary_dim=64 (partial rotary), hidden=5120.
    constexpr int NQ = 24, NKV = 4, HD = 256, ROT = 64, N_TOK = 4;
    constexpr int block_size = 16, max_blocks = 4;

    std::vector<uint16_t> hqraw((size_t)N_TOK * NQ * HD * 2);
    for (size_t i = 0; i < hqraw.size(); i++)
        hqraw[i] = bf16_bits(0.2f * (((int)((i * 2654435761u) % 2001)) - 1000) / 1000.f);
    uint16_t* dqraw = device_copy(hqraw);

    std::vector<uint16_t> hqw(HD), hkw(HD);
    for (int i = 0; i < HD; i++) { hqw[i] = bf16_bits(1.0f); hkw[i] = bf16_bits(1.0f); }
    uint16_t* dqw = device_copy(hqw);
    uint16_t* dkw = device_copy(hkw);

    bf16 *dq = nullptr, *dqgate = nullptr, *dk = nullptr;
    cudaMalloc(&dq, (size_t)N_TOK * NQ * HD * sizeof(bf16));
    cudaMalloc(&dqgate, (size_t)N_TOK * NQ * HD * sizeof(bf16));
    cudaMalloc(&dk, (size_t)N_TOK * NKV * HD * sizeof(bf16));

    std::vector<uint16_t> hv((size_t)N_TOK * NKV * HD);
    for (size_t i = 0; i < hv.size(); i++) hv[i] = bf16_bits(0.1f);
    uint16_t* dv = device_copy(hv);

    signed char *dkpool = nullptr, *dvpool = nullptr;
    cudaMalloc(&dkpool, (size_t)block_size * max_blocks * NKV * HD);
    cudaMalloc(&dvpool, (size_t)block_size * max_blocks * NKV * HD);
    __half *dkscale = nullptr, *dvscale = nullptr;
    cudaMalloc(&dkscale, (size_t)block_size * max_blocks * NKV * sizeof(__half));
    cudaMalloc(&dvscale, (size_t)block_size * max_blocks * NKV * sizeof(__half));

    std::vector<int> hblock(max_blocks);
    for (int i = 0; i < max_blocks; i++) hblock[i] = i;
    int* dblock = device_copy(hblock);
    std::vector<int> hpos(N_TOK);
    for (int i = 0; i < N_TOK; i++) hpos[i] = i;
    int* dpos = device_copy(hpos);

    // Launch and confirm it runs without CUDA error and produces finite output at this real shape.
    sparkinfer::kernels::launch_qknorm_rope_kv_partial_int8_gated(
        dqraw, dq, dqgate, dk, dv, dqw, dkw, dkpool, dvpool, dkscale, dvscale,
        dblock, dpos, N_TOK, NQ, NKV, HD, ROT, 1000000.f, 1e-6f,
        block_size, max_blocks, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("[FAIL] kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<uint16_t> hq((size_t)N_TOK * NQ * HD);
    cudaMemcpy(hq.data(), dq, hq.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    bool finite = true;
    for (uint16_t bits : hq) {
        uint32_t u = (uint32_t)bits << 16;
        float f; std::memcpy(&f, &u, sizeof(f));
        if (!std::isfinite(f)) { finite = false; break; }
    }

    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);

    // Fused path (this PR's target).
    cudaEventRecord(t0);
    for (int i = 0; i < 500; i++)
        sparkinfer::kernels::launch_qknorm_rope_kv_partial_int8_gated(
            dqraw, dq, dqgate, dk, dv, dqw, dkw, dkpool, dvpool, dkscale, dvscale,
            dblock, dpos, N_TOK, NQ, NKV, HD, ROT, 1000000.f, 1e-6f,
            block_size, max_blocks, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    // Unfused fallback: separate Q-gate split (already done by dqraw->dq/dqgate write in the
    // fused kernel; here we approximate the two-kernel cost this replaces) + rmsnorm_qk + rope.
    bf16 *dq2 = nullptr, *dk2 = nullptr;
    cudaMalloc(&dq2, (size_t)N_TOK * NQ * HD * sizeof(bf16));
    cudaMalloc(&dk2, (size_t)N_TOK * NKV * HD * sizeof(bf16));
    cudaMemcpy(dq2, dq, (size_t)N_TOK * NQ * HD * sizeof(bf16), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dk2, dk, (size_t)N_TOK * NKV * HD * sizeof(bf16), cudaMemcpyDeviceToDevice);
    cudaEventRecord(t2);
    for (int i = 0; i < 500; i++) {
        sparkinfer::kernels::launch_rmsnorm_qk(dq2, dk2, dqw, dkw, NQ, NKV, HD, 1e-6f, 0);
        sparkinfer::kernels::launch_rope(dq2, dk2, dpos, N_TOK, NQ, NKV, HD, 1000000.f, 0);
    }
    cudaEventRecord(t3);
    cudaEventSynchronize(t3);

    float ms_fused = 0, ms_unfused = 0;
    cudaEventElapsedTime(&ms_fused, t0, t1);
    cudaEventElapsedTime(&ms_unfused, t2, t3);
    std::printf("H=5120 (NQ=24,NKV=4,HD=256,ROT=64): fused %.4f ms/call, unfused (rmsnorm_qk+rope) "
                "%.4f ms/call, %.2fx, output finite=%d\n",
                ms_fused / 500, ms_unfused / 500, (ms_unfused / 500) / (ms_fused / 500), finite);
    std::printf(finite ? "[PASS] qknorm_rope_kv_partial_int8_gated runs cleanly at Qwen3.8-27B's real attention shape\n"
                        : "[FAIL] non-finite output\n");
    return finite ? 0 : 1;
}
