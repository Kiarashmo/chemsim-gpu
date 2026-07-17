#include "fingerprint.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace chemsim {

void compute_popcounts(FingerprintSet& fps) {
    fps.popcounts.resize(fps.n);
    for (uint32_t i = 0; i < fps.n; ++i) {
        const uint64_t* w = fps.fp(i);
        uint32_t pc = 0;
        for (uint32_t k = 0; k < fps.nwords; ++k)
            pc += static_cast<uint32_t>(__builtin_popcountll(w[k]));
        fps.popcounts[i] = pc;
    }
}

std::vector<uint64_t> aos_to_soa(const std::vector<uint64_t>& aos,
                                 uint32_t n, uint32_t nwords) {
    std::vector<uint64_t> soa(size_t(n) * nwords);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t w = 0; w < nwords; ++w)
            soa[size_t(w) * n + i] = aos[size_t(i) * nwords + w];
    return soa;
}

FingerprintSet load_fps(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("load_fps: cannot open " + path);

    FpsHeader h{};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in) throw std::runtime_error("load_fps: truncated header in " + path);
    if (std::memcmp(h.magic, "FPS1", 4) != 0)
        throw std::runtime_error("load_fps: bad magic in " + path);
    if (h.nwords == 0 || h.nbits != h.nwords * 64u)
        throw std::runtime_error("load_fps: inconsistent nbits/nwords in " + path);

    FingerprintSet fps;
    fps.n = h.n; fps.nwords = h.nwords; fps.nbits = h.nbits;
    fps.words.resize(size_t(h.n) * h.nwords);
    fps.popcounts.resize(h.n);

    in.read(reinterpret_cast<char*>(fps.words.data()),
            std::streamsize(fps.words.size() * sizeof(uint64_t)));
    in.read(reinterpret_cast<char*>(fps.popcounts.data()),
            std::streamsize(fps.popcounts.size() * sizeof(uint32_t)));
    if (!in) throw std::runtime_error("load_fps: truncated payload in " + path);
    return fps;
}

void save_fps(const std::string& path, const FingerprintSet& fps) {
    if (fps.words.size() != size_t(fps.n) * fps.nwords)
        throw std::runtime_error("save_fps: word buffer size mismatch");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("save_fps: cannot open " + path);

    FpsHeader h{};
    std::memcpy(h.magic, "FPS1", 4);
    h.n = fps.n; h.nwords = fps.nwords; h.nbits = fps.nbits; h.reserved = 0;
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(fps.words.data()),
              std::streamsize(fps.words.size() * sizeof(uint64_t)));
    out.write(reinterpret_cast<const char*>(fps.popcounts.data()),
              std::streamsize(fps.popcounts.size() * sizeof(uint32_t)));
    if (!out) throw std::runtime_error("save_fps: write failed for " + path);
}

}  // namespace chemsim
