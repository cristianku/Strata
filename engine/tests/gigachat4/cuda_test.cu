// Standalone; no checkpoint downloads. Exit 77 means no device test was run.
#include "../../src/gigachat4/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace strata::gigachat4;
namespace {

void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
void gpu(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

float reference_fp8(unsigned byte) {
    const unsigned e = (byte >> 3) & 15, m = byte & 7;
    const float a = e == 15 && m == 7 ? std::numeric_limits<float>::quiet_NaN() :
                    e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + m / 8.0f, int(e) - 7);
    return byte & 128 ? -a : a;
}
void check_decode(unsigned b, uint32_t decoded) {
    float got;
    std::memcpy(&got, &decoded, sizeof(got));
    const float want = reference_fp8(b);
    require((std::isnan(want) ? std::isnan(got) : want == got) && std::signbit(want) == std::signbit(got),
            "wrong E4M3FN decode at byte " + std::to_string(b));
}
__global__ void decode_all(uint32_t* out) {
    out[threadIdx.x] = strata::gigachat4::detail::e4m3fn_f32_bits(static_cast<unsigned char>(threadIdx.x));
}
void device_decode_test() {
    uint32_t host[256];
    uint32_t* dev;
    gpu(cudaMalloc(&dev, sizeof(host)), "allocate decode table");
    try {
        decode_all<<<1, 256>>>(dev);
        gpu(cudaGetLastError(), "decode table launch");
        gpu(cudaMemcpy(host, dev, sizeof(host), cudaMemcpyDeviceToHost), "read decode table");
    } catch (...) {
        cudaFree(dev);
        throw;
    }
    gpu(cudaFree(dev), "release decode table");
    for (unsigned i = 0; i < 256; ++i) check_decode(i, host[i]);
}

// Dyadic values keep these selected sums exactly representable in f32.
void fp8_tiles(MatvecBackend& backend, const std::string& key, size_t rows, size_t cols,
               MatvecBackend* compare = nullptr) {
    const size_t scale_cols = (cols + 127) / 128;
    Vec scales(((rows + 127) / 128) * scale_cols);
    for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = 0.25f * float(1 + ((3 * (i / scale_cols) + 2 * (i % scale_cols)) % 7));
    const unsigned char patterns[] = {0x00, 0x80, 0x01, 0x87, 0x08, 0xb8, 0x39, 0x42};
    std::vector<unsigned char> raw(rows * cols);
    Vec x(cols), expected(rows);
    for (size_t c = 0; c < cols; ++c) x[c] = float(int(c % 7) - 3);
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0;
        for (size_t c = 0; c < cols; ++c) {
            const size_t i = r * cols + c;
            raw[i] = patterns[(r * 3 + c) % 8];
            sum += double(reference_fp8(raw[i])) * x[c] * scales[(r / 128) * scale_cols + c / 128];
        }
        expected[r] = float(sum);
    }
    const Vec result = backend.multiply(key, raw.data(), raw.size(), "F8_E4M3", scales, rows, cols, x);
    require(result == expected, "FP8 matvec mismatch for " + key);
    if (compare) {
        const Vec other = compare->multiply(key, raw.data(), raw.size(), "F8_E4M3", scales, rows, cols, x);
        require(result.size() == other.size() && std::memcmp(result.data(), other.data(), rows * sizeof(float)) == 0,
                "serial and overlap outputs differ for " + key);
    }
}

void dense_formats(MatvecBackend& backend) {
    // Raw little-endian values: [-2, 1/2, 5/4] in BF16; include a minimum
    // nonzero half subnormal and a negative zero in the F16 input.
    const unsigned char bf[] = {0x00, 0xc0, 0x00, 0x3f, 0xa0, 0x3f};
    const unsigned char half[] = {0x01, 0x00, 0x00, 0x80, 0x00, 0xbc};
    const unsigned char fp[] = {0, 0, 0x40, 0xc0, 0, 0, 0x80, 0x3f, 0, 0, 0xa0, 0x40};
    require(backend.multiply("bf", bf, sizeof(bf), "BF16", {}, 1, 3, {3, -1, 2}) == Vec{-4.0f}, "BF16");
    require(backend.multiply("half", half, sizeof(half), "F16", {}, 1, 3, {4, 1, 0}) == Vec{0x1p-22f}, "F16 subnormal");
    require(backend.multiply("float", fp, sizeof(fp), "F32", {}, 1, 3, {1, -2, 3}) == Vec{10.0f}, "F32");
}

void cache_test() {
    // 3 + one alignment byte + 4 scale bytes per tile. Only two tiles fit.
    auto backend = make_cuda_backend(16);
    const unsigned char raw[] = {0x38, 0, 0x40};
    auto run = [&](const std::string& key, float x, float scale, bool expect_resident, bool probe = true) {
        Vec y{999};
        bool found = probe && backend->multiply_cached(key, "F8_E4M3", 1, 3, {x, 0, 2}, y);
        require(found == (probe && expect_resident), "resident probe: " + key);
        if (!found) {
            require(y == Vec{999}, "unsuccessful probe touched the destination");
            y = backend->multiply(key, raw, 3, "F8_E4M3", {scale}, 1, 3, {x, 0, 2});
        }
        require(y == Vec{(x + 4) * scale}, "cache result: " + key);
    };
    run("mlp.experts.0", 0, 1, false);         // A
    run("mlp.experts.1", 1, 2, false);         // B
    run("mlp.experts.0", 5, 1, true);          // loader hit, B now oldest
    run("dense", 3, 4, false);                 // C evicts B
    run("mlp.experts.0", 2, 1, true, false);   // direct multiply hit
    run("mlp.experts.1", 9, 2, false);         // B miss, evicts C
    const auto m = cuda_metrics(*backend);
    require(m.at("cache_hits") == 2 && m.at("cache_misses") == 4 && m.at("cache_evictions") == 2 &&
            m.at("expert_cache_hits") == 2 && m.at("expert_cache_misses") == 3, "LRU order/counters");
    require(m.at("peak_cache_bytes") == 16 && m.at("cache_bytes") == 16 &&
            m.at("streamed_tiles") == 0 && m.at("cache_entries") == 2, "cache budget accounting");
    require(m.at("weight_h2d_bytes") == 28 && m.at("h2d_bytes") == 28 + 6 * 12 &&
            m.at("d2h_bytes") == 6 * 4 && m.at("kernel_launches") == 6, "transfer accounting");
}

void bounded_stream_test() {
    auto backend = make_cuda_backend(0);
    fp8_tiles(*backend, "small-partial-blocks", 130, 139);
    fp8_tiles(*backend, "stream-crosses-row-block", 2117, 513);     // second tile starts at row 2044
    fp8_tiles(*backend, "single-row-wider-than-staging", 1, (1u << 20) + 11);
    const auto m = cuda_metrics(*backend);
    require(m.at("cache_hits") == 0 && m.at("cache_bytes") == 0 &&
            m.at("cache_misses") == 3 && m.at("streamed_tiles") == 5, "streaming tile counts");
    require(m.at("pinned_bytes") == (1u << 20) && m.at("stream_scratch_bytes").get<size_t>() < (2u << 20),
            "streaming staging bound");
}

void overlap_test() {
    auto serial = make_cuda_backend(0);
    auto pipeline = make_cuda_backend(0, true);
    fp8_tiles(*serial, "small-serial-fallback", 130, 139, pipeline.get());
    const auto before = cuda_metrics(*pipeline);
    require(before.at("overlap_enabled") == true && before.at("overlap_tiles") == 0 &&
            before.at("pinned_bytes") == (1u << 20), "small tile should not allocate an extra slot");

    // Five tiles: reuse each slot, then drain an odd tail. All four internal
    // boundaries share a partial 128-row scale block with the next tile.
    constexpr size_t rows = 8199, cols = 513;
    fp8_tiles(*serial, "five-tile-pipeline", rows, cols, pipeline.get());
    const auto used = cuda_metrics(*pipeline);
    require(used.at("overlap_calls") == 1 && used.at("overlap_tiles") == 5 &&
            used.at("kernel_launches") == 6 && used.at("streamed_tiles") == 6, "retire all pipeline tiles once");
    require(used.at("pinned_bytes") == (2u << 20) && used.at("stream_scratch_bytes") == (2u << 20),
            "double-buffer resource bound");
    const size_t transmitted = rows * cols + (((rows + 127) / 128) + 4) * ((cols + 127) / 128) * sizeof(float);
    auto delta = [&](const char* field) { return used.at(field).get<uint64_t>() - before.at(field).get<uint64_t>(); };
    require(delta("weight_h2d_bytes") == transmitted && delta("h2d_bytes") == transmitted + cols * sizeof(float) &&
            delta("d2h_bytes") == rows * sizeof(float), "pipeline transfer accounting");
    for (const auto field : {"kernel_ms", "h2d_ms", "d2h_ms"}) {
        const auto t = used.at(field).get<double>();
        require(std::isfinite(t) && t >= before.at(field).get<double>(), "pipeline event timing");
    }

    // Reuse slot 0 for a serial wide row, then run another even-tail pipeline.
    fp8_tiles(*serial, "wide-serial-fallback", 1, (1u << 20) + 11, pipeline.get());
    require(cuda_metrics(*pipeline).at("overlap_calls") == 1, "split rows must stay serial");
    fp8_tiles(*serial, "two-tile-pipeline", 2117, 513, pipeline.get());
    const auto resumed = cuda_metrics(*pipeline);
    require(resumed.at("overlap_calls") == 2 && resumed.at("overlap_tiles") == 7 &&
            resumed.at("cache_misses") == 4, "reuse pipeline after a serial call");
    require(cuda_metrics(*serial).at("overlap_enabled") == false && cuda_metrics(*serial).at("overlap_tiles") == 0,
            "overlap must be opt-in");
}

template<class F> void rejects(F f, const char* what) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(std::string("did not reject ") + what);
}
void invalid_test(MatvecBackend& b) {
    const unsigned char raw[] = {0, 0, 0, 0};
    rejects([&]{ b.multiply("a", raw, 1, "F8_E4M3", {}, 1, 1, {1}); }, "missing scale");
    rejects([&]{ b.multiply("a", raw, 2, "BF16", {1}, 1, 1, {1}); }, "unexpected dense scale");
    rejects([&]{ b.multiply("a", raw, 3, "F32", {}, 1, 1, {1}); }, "truncated raw");
    rejects([&]{ b.multiply("a", raw, 4, "F32", {}, 1, 1, {}); }, "wrong activation size");
    rejects([&]{ b.multiply("a", raw, 4, "FNUZ", {}, 1, 1, {1}); }, "unknown dtype");
    rejects([&]{ b.multiply("a", raw, 4, "F32", {}, std::numeric_limits<size_t>::max(), 2, {1, 1}); }, "overflow");
}
} // namespace

int main() try {
    for (unsigned i = 0; i < 256; ++i)
        check_decode(i, strata::gigachat4::detail::e4m3fn_f32_bits(static_cast<unsigned char>(i)));
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices)) {
        std::fprintf(stderr, "SKIP device tests: no CUDA device/driver (%s)\n", cudaGetErrorString(status));
        return 77;
    }
    gpu(status, "enumerate devices");
    int device = 0;
    cudaDeviceProp prop{};
    for (; device < devices; ++device) {
        gpu(cudaGetDeviceProperties(&prop, device), "device properties");
        if (prop.major >= 7) break;
    }
    if (device == devices) {
        std::fprintf(stderr, "SKIP device tests: need compute capability >= 7.0\n");
        return 77;
    }
    gpu(cudaSetDevice(device), "select device");
    device_decode_test();
    auto backend = make_cuda_backend(1u << 20);
    fp8_tiles(*backend, "resident-partial-blocks", 133, 139);
    dense_formats(*backend);
    invalid_test(*backend);
    cache_test();
    bounded_stream_test();
    overlap_test();
    std::printf("PASS: CUDA matvec, FP8 decode, serial/overlap streaming and LRU on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
