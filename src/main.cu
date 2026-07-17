// Driver and benchmark harness.
// Loads fingerprints, runs a stage (or all), times it, optionally checks it
// against the CPU reference, and can print top-k or thresholded hits.
// Stages 7 (streams) and 8 (result handling) live here since they are host work.
//
// Examples:
//   ./chemsim --lib data/lib.fps --stage all --validate
//   ./chemsim --query data/q.fps --lib data/lib.fps --stage 4 --repeat 20
//   ./chemsim --lib data/lib.fps --stage 3 --stream 4
//   ./chemsim --query data/q.fps --lib data/lib.fps --stage 3 --topk 10 --out hits.tsv
#include "fingerprint.hpp"
#include "kernels.cuh"
#include "tanimoto_cpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace chemsim;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

struct Args {
    std::string query;                 // empty means all-pairs (query = lib)
    std::string lib;
    std::string out;
    int   stage     = 0;
    bool  all       = true;
    bool  validate  = false;
    int   block     = 16;
    int   warps     = 8;               // warps per block for stage 6
    int   repeat    = 10;
    int   stream    = 0;               // number of streams for stage 7
    int   nq_cap    = 0;               // 0 means no cap
    int   nl_cap    = 0;
    int   topk      = 0;
    float threshold = -1.0f;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
        "usage: chemsim --lib FILE [--query FILE] [--stage N|all] [--validate]\n"
        "               [--block B] [--warps W] [--repeat R] [--stream N]\n"
        "               [--nq N] [--nl N] [--topk K] [--threshold T] [--out FILE]\n");
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if      (s == "--query")     a.query = next();
        else if (s == "--lib")       a.lib = next();
        else if (s == "--out")       a.out = next();
        else if (s == "--validate")  a.validate = true;
        else if (s == "--block")     a.block = std::stoi(next());
        else if (s == "--warps")     a.warps = std::stoi(next());
        else if (s == "--repeat")    a.repeat = std::stoi(next());
        else if (s == "--stream")    a.stream = std::stoi(next());
        else if (s == "--nq")        a.nq_cap = std::stoi(next());
        else if (s == "--nl")        a.nl_cap = std::stoi(next());
        else if (s == "--topk")      a.topk = std::stoi(next());
        else if (s == "--threshold") a.threshold = std::stof(next());
        else if (s == "--stage") {
            std::string v = next();
            if (v == "all") { a.all = true; a.stage = 0; }
            else            { a.all = false; a.stage = std::stoi(v); }
        } else usage();
    }
    if (a.lib.empty()) usage();
    return a;
}

// Keep only the first `cap` fingerprints (cap 0 leaves it unchanged).
FingerprintSet cap_set(FingerprintSet s, int cap) {
    if (cap > 0 && uint32_t(cap) < s.n) {
        s.words.resize(size_t(cap) * s.nwords);
        s.popcounts.resize(cap);
        s.n = cap;
    }
    return s;
}

double max_abs_err(const std::vector<float>& gpu, const std::vector<float>& ref) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        m = std::max(m, double(std::abs(gpu[i] - ref[i])));
    return m;
}

struct DeviceData {
    // Both layouts: AoS for stages 1,2,6 and SoA for stages 3,4,5.
    uint64_t *q_aos = nullptr, *l_aos = nullptr;
    uint64_t *q_soa = nullptr, *l_soa = nullptr;
    uint32_t *qpc = nullptr, *lpc = nullptr;
    float    *out = nullptr;
    uint32_t nq = 0, nl = 0, nwords = 0;
};

void upload(const FingerprintSet& q, const FingerprintSet& l, DeviceData& d) {
    d.nq = q.n; d.nl = l.n; d.nwords = q.nwords;
    const size_t qw = size_t(q.n) * q.nwords, lw = size_t(l.n) * l.nwords;

    auto q_soa = aos_to_soa(q.words, q.n, q.nwords);
    auto l_soa = aos_to_soa(l.words, l.n, l.nwords);

    CUDA_CHECK(cudaMalloc(&d.q_aos, qw * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.l_aos, lw * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.q_soa, qw * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.l_soa, lw * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.qpc, q.n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.lpc, l.n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.out, size_t(q.n) * l.n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d.q_aos, q.words.data(), qw * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.l_aos, l.words.data(), lw * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.q_soa, q_soa.data(), qw * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.l_soa, l_soa.data(), lw * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.qpc, q.popcounts.data(), q.n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.lpc, l.popcounts.data(), l.n * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void free_device(DeviceData& d) {
    cudaFree(d.q_aos); cudaFree(d.l_aos); cudaFree(d.q_soa); cudaFree(d.l_soa);
    cudaFree(d.qpc); cudaFree(d.lpc); cudaFree(d.out);
}

void launch(int stage, const Args& a, DeviceData& d) {
    switch (stage) {
        case 1: run_stage1_naive (d.q_aos, d.qpc, d.nq, d.l_aos, d.lpc, d.nl, d.nwords, d.out, a.block); break;
        case 2: run_stage2_popcll(d.q_aos, d.qpc, d.nq, d.l_aos, d.lpc, d.nl, d.nwords, d.out, a.block); break;
        case 3: run_stage3_soa   (d.q_soa, d.qpc, d.nq, d.l_soa, d.lpc, d.nl, d.nwords, d.out); break;
        case 4: run_stage4_tiled (d.q_soa, d.qpc, d.nq, d.l_soa, d.lpc, d.nl, d.nwords, d.out); break;
        case 5: run_stage5_coarsen(d.q_soa, d.qpc, d.nq, d.l_soa, d.lpc, d.nl, d.nwords, d.out); break;
        case 6: run_stage6_warp  (d.q_aos, d.qpc, d.nq, d.l_aos, d.lpc, d.nl, d.nwords, d.out, a.warps); break;
        default: std::fprintf(stderr, "unknown stage %d\n", stage); std::exit(2);
    }
}

// 3 warmups, then `repeat` timed runs. GCUPS = billion cells per second.
void benchmark(int stage, const Args& a, DeviceData& d) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    for (int w = 0; w < 3; ++w) launch(stage, a, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f, sum = 0.0f;
    for (int r = 0; r < a.repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(beg));
        launch(stage, a, d);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        best = std::min(best, ms);
        sum += ms;
    }
    const double cells = double(d.nq) * double(d.nl);
    const double gcups = cells / (best * 1e-3) / 1e9;
    std::printf("stage %d  %-38s  min %8.3f ms  avg %8.3f ms  %7.2f GCUPS\n",
                stage, stage_name(stage), best, sum / a.repeat, gcups);
    cudaEventDestroy(beg); cudaEventDestroy(end);
}

// Stage 7. Split query rows into chunks and overlap each chunk's host-to-device
// copy with the kernel on other chunks, using several streams.
void run_streamed(const FingerprintSet& q, const Args& a, DeviceData& d) {
    const int nstreams = std::max(1, a.stream);
    const uint32_t chunk = (d.nq + nstreams - 1) / nstreams;

    std::vector<cudaStream_t> streams(nstreams);
    for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

    // Pinned host copy so the async transfers can overlap.
    uint64_t* h_pinned = nullptr;
    const size_t qwords = size_t(d.nq) * d.nwords;
    CUDA_CHECK(cudaMallocHost(&h_pinned, qwords * sizeof(uint64_t)));
    std::memcpy(h_pinned, q.words.data(), qwords * sizeof(uint64_t));

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg)); CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int s = 0; s < nstreams; ++s) {
        const uint32_t r0 = s * chunk;
        if (r0 >= d.nq) break;
        const uint32_t rows = std::min(chunk, d.nq - r0);
        uint64_t* q_dst = d.q_aos + size_t(r0) * d.nwords;
        CUDA_CHECK(cudaMemcpyAsync(q_dst, h_pinned + size_t(r0) * d.nwords,
                                   size_t(rows) * d.nwords * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, streams[s]));
        dim3 b(a.block, a.block);
        dim3 g((d.nl + b.x - 1) / b.x, (rows + b.y - 1) / b.y);
        run_stage2_popcll(q_dst, d.qpc + r0, rows, d.l_aos, d.lpc, d.nl,
                          d.nwords, d.out + size_t(r0) * d.nl, a.block);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    std::printf("stage 7  CUDA streams (%d) overlap copy+compute      total %8.3f ms\n",
                nstreams, ms);

    for (auto& s : streams) cudaStreamDestroy(s);
    cudaFreeHost(h_pinned);
    cudaEventDestroy(beg); cudaEventDestroy(end);
}

// Stage 8. Copy the matrix back and print either top-k per query or all pairs
// above the threshold, so we never hand the user the full matrix.
void emit_results(const Args& a, DeviceData& d) {
    std::vector<float> mat(size_t(d.nq) * d.nl);
    CUDA_CHECK(cudaMemcpy(mat.data(), d.out, mat.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::ofstream fout;
    std::ostream* os = &std::cout;
    if (!a.out.empty()) { fout.open(a.out); os = &fout; }
    *os << "query\tlib\ttanimoto\n";

    for (uint32_t i = 0; i < d.nq; ++i) {
        const float* row = mat.data() + size_t(i) * d.nl;
        if (a.topk > 0) {
            std::vector<uint32_t> idx(d.nl);
            for (uint32_t j = 0; j < d.nl; ++j) idx[j] = j;
            const int k = std::min<int>(a.topk, d.nl);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](uint32_t x, uint32_t y) { return row[x] > row[y]; });
            for (int r = 0; r < k; ++r)
                *os << i << '\t' << idx[r] << '\t' << row[idx[r]] << '\n';
        } else {
            for (uint32_t j = 0; j < d.nl; ++j)
                if (row[j] >= a.threshold)
                    *os << i << '\t' << j << '\t' << row[j] << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    FingerprintSet lib = cap_set(load_fps(a.lib), a.nl_cap);
    FingerprintSet query = a.query.empty() ? lib : cap_set(load_fps(a.query), a.nq_cap);
    if (a.query.empty() && a.nq_cap > 0) query = cap_set(query, a.nq_cap);

    if (query.nwords != lib.nwords) {
        std::fprintf(stderr, "fingerprint length mismatch: query %u vs lib %u words\n",
                     query.nwords, lib.nwords);
        return 1;
    }
    std::printf("query %u x lib %u  (%u words/fp = %u bits)\n",
                query.n, lib.n, query.nwords, query.nbits);

    DeviceData d{};
    upload(query, lib, d);

    std::vector<float> ref;
    if (a.validate) {
        ref.assign(size_t(query.n) * lib.n, 0.0f);
        tanimoto_matrix_cpu(query, lib, ref.data());
    }

    if (a.stream > 0) { run_streamed(query, a, d); }

    const int lo = a.all ? 1 : a.stage;
    const int hi = a.all ? 6 : a.stage;
    if (!(a.stream > 0 && !a.all)) {
        for (int s = lo; s <= hi; ++s) {
            if (s > 6) break;   // stages 7 and 8 use --stream and --topk/--threshold
            launch(s, a, d);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (a.validate) {
                std::vector<float> got(ref.size());
                CUDA_CHECK(cudaMemcpy(got.data(), d.out, got.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                double err = max_abs_err(got, ref);
                std::printf("  [validate] stage %d max diff vs CPU = %.3e  %s\n",
                            s, err, err <= 1e-6 ? "OK" : "FAIL");
            }
            benchmark(s, a, d);
        }
    }

    if (a.topk > 0 || a.threshold >= 0.0f) {
        launch(3, a, d);        // fill the matrix with a correct stage first
        CUDA_CHECK(cudaDeviceSynchronize());
        emit_results(a, d);
    }

    free_device(d);
    return 0;
}
