// Stage 0: single-threaded CPU reference. Every GPU stage is checked against it.
#pragma once

#include "fingerprint.hpp"

#include <cstdint>
#include <vector>

namespace chemsim {

// T = c / (a + b - c), with T(0,0) = 0.
float tanimoto(const uint64_t* a, const uint64_t* b, uint32_t nwords);

// Same, with the popcounts pa, pb passed in.
float tanimoto_pc(const uint64_t* a, const uint64_t* b, uint32_t nwords,
                  uint32_t pa, uint32_t pb);

// Dense similarity matrix, row-major [nq x nlib]. out needs nq*nlib floats.
void tanimoto_matrix_cpu(const FingerprintSet& query,
                         const FingerprintSet& lib,
                         float* out);

}  // namespace chemsim
