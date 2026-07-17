#include "kernels.cuh"

#include <cuda_runtime.h>

namespace chemsim {

// Software popcount, used only by the naive stage 1.
__device__ __forceinline__ uint32_t popc64_manual(uint64_t x) {
    uint32_t c = 0;
    while (x) { x &= (x - 1); ++c; }
    return c;
}

__device__ __forceinline__ float tanimoto_from_c(uint32_t c, uint32_t pa, uint32_t pb) {
    const uint32_t denom = pa + pb - c;
    return denom == 0u ? 0.0f : float(c) / float(denom);
}

const char* stage_name(int stage) {
    switch (stage) {
        case 1: return "naive (AoS, manual popcount)";
        case 2: return "bit-packed + __popcll (AoS)";
        case 3: return "memory coalescing (SoA)";
        case 4: return "shared-memory tiling";
        case 5: return "thread coarsening / register blocking";
        case 6: return "warp-shuffle reduction";
        default: return "unknown";
    }
}

// Stage 1
__global__ void k_stage1(const uint64_t* q, const uint32_t* qpc, uint32_t nq,
                         const uint64_t* l, const uint32_t* lpc, uint32_t nl,
                         uint32_t nwords, float* out) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nq || j >= nl) return;

    const uint64_t* a = q + size_t(i) * nwords;
    const uint64_t* b = l + size_t(j) * nwords;
    uint32_t c = 0;
    for (uint32_t k = 0; k < nwords; ++k)
        c += popc64_manual(a[k] & b[k]);
    out[size_t(i) * nl + j] = tanimoto_from_c(c, qpc[i], lpc[j]);
}

// Stage 2
__global__ void k_stage2(const uint64_t* q, const uint32_t* qpc, uint32_t nq,
                         const uint64_t* l, const uint32_t* lpc, uint32_t nl,
                         uint32_t nwords, float* out) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nq || j >= nl) return;

    const uint64_t* a = q + size_t(i) * nwords;
    const uint64_t* b = l + size_t(j) * nwords;
    uint32_t c = 0;
    for (uint32_t k = 0; k < nwords; ++k)
        c += __popcll(a[k] & b[k]);
    out[size_t(i) * nl + j] = tanimoto_from_c(c, qpc[i], lpc[j]);
}

// Stage 3. SoA, so threads with consecutive j read consecutive addresses.
__global__ void k_stage3(const uint64_t* q, const uint32_t* qpc, uint32_t nq,
                         const uint64_t* l, const uint32_t* lpc, uint32_t nl,
                         uint32_t nwords, float* out) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nq || j >= nl) return;

    uint32_t c = 0;
    for (uint32_t k = 0; k < nwords; ++k)
        c += __popcll(q[size_t(k) * nq + i] & l[size_t(k) * nl + j]);
    out[size_t(i) * nl + j] = tanimoto_from_c(c, qpc[i], lpc[j]);
}

// Stage 4
#define TILE 16      // TILE x TILE output block per thread block
#define MAXW 64      // max words per fingerprint the tiled kernel supports

__global__ void k_stage4(const uint64_t* q, const uint32_t* qpc, uint32_t nq,
                         const uint64_t* l, const uint32_t* lpc, uint32_t nl,
                         uint32_t nwords, float* out) {
    __shared__ uint64_t sQ[TILE * MAXW];
    __shared__ uint64_t sL[TILE * MAXW];

    const uint32_t rowBase = blockIdx.y * TILE;
    const uint32_t colBase = blockIdx.x * TILE;
    const uint32_t tid     = threadIdx.y * TILE + threadIdx.x;
    const uint32_t nthreads = TILE * TILE;

    // Load both tiles into shared memory (coalesced: local varies fastest).
    for (uint32_t e = tid; e < TILE * nwords; e += nthreads) {
        const uint32_t local = e % TILE;
        const uint32_t k     = e / TILE;
        const uint32_t gq = rowBase + local;
        const uint32_t gl = colBase + local;
        sQ[k * TILE + local] = (gq < nq) ? q[size_t(k) * nq + gq] : 0ull;
        sL[k * TILE + local] = (gl < nl) ? l[size_t(k) * nl + gl] : 0ull;
    }
    __syncthreads();

    const uint32_t i = rowBase + threadIdx.y;
    const uint32_t j = colBase + threadIdx.x;
    if (i < nq && j < nl) {
        uint32_t c = 0;
        for (uint32_t k = 0; k < nwords; ++k)
            c += __popcll(sQ[k * TILE + threadIdx.y] & sL[k * TILE + threadIdx.x]);
        out[size_t(i) * nl + j] = tanimoto_from_c(c, qpc[i], lpc[j]);
    }
}

// Stage 5
#define COARSE 4     // library columns per thread

__global__ void k_stage5(const uint64_t* q, const uint32_t* qpc, uint32_t nq,
                         const uint64_t* l, const uint32_t* lpc, uint32_t nl,
                         uint32_t nwords, float* out) {
    const uint32_t i    = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t j0   = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE;
    if (i >= nq) return;

    uint32_t acc[COARSE];
#pragma unroll
    for (int t = 0; t < COARSE; ++t) acc[t] = 0;

    for (uint32_t k = 0; k < nwords; ++k) {
        const uint64_t qw = q[size_t(k) * nq + i];   // load once, reuse across columns
        const uint64_t* lk = l + size_t(k) * nl;
#pragma unroll
        for (int t = 0; t < COARSE; ++t) {
            const uint32_t j = j0 + t;
            if (j < nl) acc[t] += __popcll(qw & lk[j]);
        }
    }
#pragma unroll
    for (int t = 0; t < COARSE; ++t) {
        const uint32_t j = j0 + t;
        if (j < nl) out[size_t(i) * nl + j] = tanimoto_from_c(acc[t], qpc[i], lpc[j]);
    }
}

// Stage 6
__global__ void k_stage6(const uint64_t* q, const uint32_t* qpc, uint32_t nq,
                         const uint64_t* l, const uint32_t* lpc, uint32_t nl,
                         uint32_t nwords, float* out) {
    const uint32_t lane   = threadIdx.x;
    const uint32_t warpId = threadIdx.y;
    const uint32_t j = blockIdx.x;
    const uint32_t i = (blockIdx.y * blockDim.y + warpId);
    if (i >= nq || j >= nl) return;

    const uint64_t* a = q + size_t(i) * nwords;
    const uint64_t* b = l + size_t(j) * nwords;
    uint32_t c = 0;
    for (uint32_t k = lane; k < nwords; k += 32)   // lanes split the words
        c += __popcll(a[k] & b[k]);

    for (int off = 16; off > 0; off >>= 1)          // sum across the warp
        c += __shfl_down_sync(0xffffffffu, c, off);

    if (lane == 0)
        out[size_t(i) * nl + j] = tanimoto_from_c(c, qpc[i], lpc[j]);
}

// Host launchers

static dim3 grid2d(uint32_t nl, uint32_t nq, dim3 block) {
    return dim3((nl + block.x - 1) / block.x, (nq + block.y - 1) / block.y);
}

void run_stage1_naive(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                      const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                      uint32_t nwords, float* d_out, int block) {
    dim3 b(block, block);
    k_stage1<<<grid2d(nl, nq, b), b>>>(d_q, d_qpc, nq, d_l, d_lpc, nl, nwords, d_out);
}

void run_stage2_popcll(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                       const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                       uint32_t nwords, float* d_out, int block) {
    dim3 b(block, block);
    k_stage2<<<grid2d(nl, nq, b), b>>>(d_q, d_qpc, nq, d_l, d_lpc, nl, nwords, d_out);
}

void run_stage3_soa(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                    const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                    uint32_t nwords, float* d_out) {
    dim3 b(32, 8);
    k_stage3<<<grid2d(nl, nq, b), b>>>(d_q, d_qpc, nq, d_l, d_lpc, nl, nwords, d_out);
}

void run_stage4_tiled(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                      const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                      uint32_t nwords, float* d_out) {
    dim3 b(TILE, TILE);
    dim3 g((nl + TILE - 1) / TILE, (nq + TILE - 1) / TILE);
    k_stage4<<<g, b>>>(d_q, d_qpc, nq, d_l, d_lpc, nl, nwords, d_out);
}

void run_stage5_coarsen(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                        const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                        uint32_t nwords, float* d_out) {
    dim3 b(32, 8);
    dim3 g((nl + (b.x * COARSE) - 1) / (b.x * COARSE), (nq + b.y - 1) / b.y);
    k_stage5<<<g, b>>>(d_q, d_qpc, nq, d_l, d_lpc, nl, nwords, d_out);
}

void run_stage6_warp(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                     const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                     uint32_t nwords, float* d_out, int warps_per_block) {
    dim3 b(32, warps_per_block);
    dim3 g(nl, (nq + warps_per_block - 1) / warps_per_block);
    k_stage6<<<g, b>>>(d_q, d_qpc, nq, d_l, d_lpc, nl, nwords, d_out);
}

}  // namespace chemsim
