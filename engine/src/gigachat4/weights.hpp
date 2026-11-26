#pragma once

#include "model.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace strata::gigachat4 {

class MatvecBackend {
public:
    virtual ~MatvecBackend() = default;
    // Before host I/O, optionally multiply a resident tile by the current x.
    // A hit must fill result with rows floats. A miss falls through to multiply().
    virtual bool multiply_cached(const std::string& /*key*/, const std::string& /*dtype*/,
                                 size_t /*rows*/, size_t /*cols*/, const Vec& /*x*/, Vec& /*result*/) {
        return false;
    }
    // Synchronous. Pointers/references are valid only for this call. key identifies
    // this reader, tensor and half-open row range. Raw data is little-endian.
    // FP8: F8_E4M3, scales[local_row/128, col/128], rounded-up block counts.
    // Dense: BF16, F16 or F32, empty scales. Return exactly rows floats.
    virtual Vec multiply(const std::string& key, const unsigned char* raw, size_t nbytes,
                         const std::string& dtype, const Vec& scales, size_t rows,
                         size_t cols, const Vec& x) = 0;
};

struct WeightStats {
    size_t tensors = 0, shards = 0;
    uint64_t read_ops = 0, bytes_read = 0; // Payload I/O only; excludes headers/index.
    double payload_read_seconds = 0;
    uint64_t cache_hits = 0, cache_misses = 0, evictions = 0;
    size_t cache_bytes = 0, peak_cache_bytes = 0, max_read_bytes = 0;
    bool prefetch_supported = false;
    uint64_t prefetch_requests = 0, prefetch_bytes = 0, prefetch_errors = 0;
};

// Immutable checkpoint for the lifetime of the reader. Single calling thread.
// A supplied backend is non-owning and must outlive the reader.
class FileWeights final : public Weights {
public:
    explicit FileWeights(const std::filesystem::path& dir, size_t cache_bytes = 0,
                         MatvecBackend* backend = nullptr);
    ~FileWeights() override;
    FileWeights(const FileWeights&) = delete;
    FileWeights& operator=(const FileWeights&) = delete;

    Vec linear(const std::string& name, const Vec& x, size_t out) override;
    Vec read(const std::string& name, const std::vector<size_t>& shape) override;
    Vec row(const std::string& name, size_t rows, size_t cols, size_t index) override;
    void prefetch_expert(const std::string& expert_prefix) override;
    WeightStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::gigachat4
