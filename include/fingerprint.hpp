// Packed fingerprints and the .fps file format.
// Plain C++17, shared by the CUDA engine, CPU reference, and tests.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace chemsim {

// File header, little-endian.
struct FpsHeader {
    char     magic[4];   // "FPS1"
    uint32_t n;          // number of fingerprints
    uint32_t nwords;     // 64-bit words per fingerprint (nbits / 64)
    uint32_t nbits;      // fingerprint bit length, e.g. 1024 or 2048
    uint32_t reserved;   // 0
};

// Fingerprints in AoS layout: words has n*nwords entries, popcounts has n.
struct FingerprintSet {
    uint32_t n      = 0;
    uint32_t nwords = 0;
    uint32_t nbits  = 0;
    std::vector<uint64_t> words;
    std::vector<uint32_t> popcounts;

    const uint64_t* fp(uint32_t i) const { return words.data() + size_t(i) * nwords; }
    uint64_t*       fp(uint32_t i)       { return words.data() + size_t(i) * nwords; }
};

void compute_popcounts(FingerprintSet& fps);

// AoS to SoA: soa[w*n + i] == aos[i*nwords + w]. Used by stages 3+.
std::vector<uint64_t> aos_to_soa(const std::vector<uint64_t>& aos,
                                 uint32_t n, uint32_t nwords);

FingerprintSet load_fps(const std::string& path);
void           save_fps(const std::string& path, const FingerprintSet& fps);

}  // namespace chemsim
