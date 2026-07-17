// The optimization ladder (stages 1-6) and their host launchers.
// Each launcher fills a dense Tanimoto matrix d_out[nq x nlib], row-major float.
// Layouts:
//   AoS: word k of fp i at words[i*nwords + k]
//   SoA: word k of fp i at words[k*n + i]
#pragma once

#include <cstdint>

namespace chemsim {

const char* stage_name(int stage);

// Stage 1: one thread per cell, software popcount. AoS. The slow baseline.
void run_stage1_naive(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                      const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                      uint32_t nwords, float* d_out, int block);

// Stage 2: same as stage 1 but with the __popcll intrinsic. AoS.
void run_stage2_popcll(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                       const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                       uint32_t nwords, float* d_out, int block);

// Stage 3: SoA layout so a warp reads adjacent addresses (coalesced). SoA.
void run_stage3_soa(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                    const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                    uint32_t nwords, float* d_out);

// Stage 4: load a tile of fps into shared memory once, reuse on chip. SoA.
void run_stage4_tiled(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                      const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                      uint32_t nwords, float* d_out);

// Stage 5: each thread does COARSE columns, reusing the query word. SoA.
void run_stage5_coarsen(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                        const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                        uint32_t nwords, float* d_out);

// Stage 6: one warp per cell, lanes split the words and reduce with shuffle. AoS.
void run_stage6_warp(const uint64_t* d_q, const uint32_t* d_qpc, uint32_t nq,
                     const uint64_t* d_l, const uint32_t* d_lpc, uint32_t nl,
                     uint32_t nwords, float* d_out, int warps_per_block);

}  // namespace chemsim
