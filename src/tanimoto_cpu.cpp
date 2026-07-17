#include "tanimoto_cpu.hpp"

#include <stdexcept>

namespace chemsim {

float tanimoto_pc(const uint64_t* a, const uint64_t* b, uint32_t nwords,
                  uint32_t pa, uint32_t pb) {
    uint32_t c = 0;
    for (uint32_t k = 0; k < nwords; ++k)
        c += static_cast<uint32_t>(__builtin_popcountll(a[k] & b[k]));
    const uint32_t denom = pa + pb - c;   // union size
    if (denom == 0) return 0.0f;          // both empty, T = 0
    return static_cast<float>(c) / static_cast<float>(denom);
}

float tanimoto(const uint64_t* a, const uint64_t* b, uint32_t nwords) {
    uint32_t pa = 0, pb = 0;
    for (uint32_t k = 0; k < nwords; ++k) {
        pa += static_cast<uint32_t>(__builtin_popcountll(a[k]));
        pb += static_cast<uint32_t>(__builtin_popcountll(b[k]));
    }
    return tanimoto_pc(a, b, nwords, pa, pb);
}

void tanimoto_matrix_cpu(const FingerprintSet& query,
                         const FingerprintSet& lib,
                         float* out) {
    if (query.nwords != lib.nwords)
        throw std::runtime_error("tanimoto_matrix_cpu: fingerprint length mismatch");
    const uint32_t nw = query.nwords;
    for (uint32_t i = 0; i < query.n; ++i) {
        const uint64_t* qa = query.fp(i);
        const uint32_t  pa = query.popcounts[i];
        float* row = out + size_t(i) * lib.n;
        for (uint32_t j = 0; j < lib.n; ++j)
            row[j] = tanimoto_pc(qa, lib.fp(j), nw, pa, lib.popcounts[j]);
    }
}

}  // namespace chemsim
