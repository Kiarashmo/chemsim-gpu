// CPU-only tests: the Tanimoto formula, edge cases, .fps I/O, the AoS/SoA
// layout, and the SoA index math the kernels rely on. No GPU needed.
#include "fingerprint.hpp"
#include "tanimoto_cpu.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace chemsim;

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; }             \
        else         { std::printf("ok:   %s\n", msg); }                       \
    } while (0)

static bool close(float a, float b) { return std::fabs(a - b) <= 1e-6f; }

// Build a set of `n` fingerprints of `nwords` words from explicit bit lists.
static FingerprintSet make_set(uint32_t nwords,
                               const std::vector<std::vector<int>>& bits) {
    FingerprintSet s;
    s.n = uint32_t(bits.size());
    s.nwords = nwords;
    s.nbits = nwords * 64;
    s.words.assign(size_t(s.n) * nwords, 0ull);
    for (uint32_t i = 0; i < s.n; ++i)
        for (int bit : bits[i])
            s.fp(i)[bit / 64] |= (1ull << (bit % 64));
    compute_popcounts(s);
    return s;
}

// Host mirror of the SoA kernels (stages 3/4/5): reads from SoA buffers.
static float tanimoto_soa(const std::vector<uint64_t>& qs, uint32_t nq, uint32_t i,
                          const std::vector<uint64_t>& ls, uint32_t nl, uint32_t j,
                          uint32_t nwords, uint32_t pa, uint32_t pb) {
    uint32_t c = 0;
    for (uint32_t k = 0; k < nwords; ++k)
        c += __builtin_popcountll(qs[size_t(k) * nq + i] & ls[size_t(k) * nl + j]);
    uint32_t denom = pa + pb - c;
    return denom == 0 ? 0.0f : float(c) / float(denom);
}

int main() {
    // --- edge cases -----------------------------------------------------
    {
        auto s = make_set(2, {
            {},              // empty
            {0, 1, 2},       // A
            {1, 2, 3},       // B
            {0, 1, 2},       // == A
            {64, 65},        // disjoint from A (different word)
        });
        CHECK(close(tanimoto(s.fp(0), s.fp(0), 2), 0.0f), "T(empty,empty)=0");
        CHECK(close(tanimoto(s.fp(1), s.fp(3), 2), 1.0f), "T(A,A)=1");
        CHECK(close(tanimoto(s.fp(1), s.fp(2), 2), 0.5f), "T(A,B)=0.5 (c=2,denom=4)");
        CHECK(close(tanimoto(s.fp(1), s.fp(4), 2), 0.0f), "T(A,disjoint)=0");
        CHECK(s.popcounts[1] == 3 && s.popcounts[4] == 2, "popcounts correct");
    }

    // --- AoS <-> SoA roundtrip and matching similarities ----------------
    {
        auto s = make_set(3, {{0,5,70}, {5,70,130}, {1,2,3,4}});
        auto soa = aos_to_soa(s.words, s.n, s.nwords);
        bool layout_ok = true;
        for (uint32_t i = 0; i < s.n; ++i)
            for (uint32_t k = 0; k < s.nwords; ++k)
                if (soa[size_t(k) * s.n + i] != s.words[size_t(i) * s.nwords + k])
                    layout_ok = false;
        CHECK(layout_ok, "aos_to_soa places word k of fp i at k*n+i");

        bool sim_ok = true;
        for (uint32_t i = 0; i < s.n; ++i)
            for (uint32_t j = 0; j < s.n; ++j) {
                float ref = tanimoto_pc(s.fp(i), s.fp(j), s.nwords,
                                        s.popcounts[i], s.popcounts[j]);
                float via = tanimoto_soa(soa, s.n, i, soa, s.n, j, s.nwords,
                                         s.popcounts[i], s.popcounts[j]);
                if (!close(ref, via)) sim_ok = false;
            }
        CHECK(sim_ok, "SoA-indexed similarity matches AoS reference (all pairs)");
    }

    // --- matrix vs pairwise ---------------------------------------------
    {
        auto q = make_set(2, {{0,1}, {64}});
        auto l = make_set(2, {{0,1,2}, {64,65}, {}});
        std::vector<float> mat(size_t(q.n) * l.n);
        tanimoto_matrix_cpu(q, l, mat.data());
        bool ok = true;
        for (uint32_t i = 0; i < q.n; ++i)
            for (uint32_t j = 0; j < l.n; ++j)
                if (!close(mat[i * l.n + j],
                           tanimoto_pc(q.fp(i), l.fp(j), q.nwords,
                                       q.popcounts[i], l.popcounts[j]))) ok = false;
        CHECK(ok, "tanimoto_matrix_cpu matches element-wise");
    }

    // --- .fps I/O roundtrip ---------------------------------------------
    {
        auto s = make_set(4, {{0,1,63,64,255}, {}, {100,200,250}});
        std::string path = "/tmp/chemsim_test.fps";
        save_fps(path, s);
        auto r = load_fps(path);
        bool ok = r.n == s.n && r.nwords == s.nwords && r.nbits == s.nbits &&
                  r.words == s.words && r.popcounts == s.popcounts;
        CHECK(ok, ".fps save/load roundtrip preserves data");
        std::remove(path.c_str());
    }

    if (g_fail == 0) { std::printf("\nALL TESTS PASSED\n"); return 0; }
    std::printf("\n%d TEST(S) FAILED\n", g_fail);
    return 1;
}
