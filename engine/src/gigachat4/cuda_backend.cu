#include "cuda_backend.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace strata::gigachat4 {
namespace {

constexpr size_t block_side = 128;             // checkpoint scale grid
constexpr size_t stage_bytes = 1024 * 1024;    // pinned host buffer; also max streamed raw tile
constexpr unsigned threads = 256, warp = 32;

void check(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("GigaChat4 CUDA: ") + where + ": " + cudaGetErrorString(err));
}

size_t mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::invalid_argument("GigaChat4 CUDA: tensor size overflow");
    return a * b;
}
size_t add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::invalid_argument("GigaChat4 CUDA: tensor size overflow");
    return a + b;
}
size_t div_up(size_t n, size_t d) { return n / d + (n % d != 0); }

enum class Format { fp8, bf16, fp16, fp32 };
Format parse(const std::string& name) {
    if (name == "F8_E4M3") return Format::fp8;
    if (name == "BF16") return Format::bf16;
    if (name == "F16") return Format::fp16;
    if (name == "F32") return Format::fp32;
    throw std::invalid_argument("GigaChat4 CUDA: unsupported dtype: " + name);
}
size_t width(Format f) { return f == Format::fp8 ? 1 : f == Format::fp32 ? 4 : 2; }

// One CUDA allocation per resident tile. Account for padding before the f32 scales.
struct Layout {
    size_t raw, scale_offset, scale_bytes, total;
    Layout(size_t raw_bytes, size_t nscale)
        : raw(raw_bytes), scale_offset(nscale ? add(raw, (4 - raw % 4) % 4) : raw),
          scale_bytes(mul(nscale, sizeof(float))), total(add(scale_offset, scale_bytes)) {}
};

Layout describe(Format f, size_t rows, size_t cols, const Vec& x) {
    if (!rows || !cols || x.size() != cols)
        throw std::invalid_argument("GigaChat4 CUDA: invalid matvec dimensions");
    const size_t ns = f == Format::fp8 ? mul(div_up(rows, block_side), div_up(cols, block_side)) : 0;
    return Layout(mul(mul(rows, cols), width(f)), ns);
}

// Each warp owns a row. The termwise scale and accumulation are FP32 on all
// architectures; the file representation is never expanded into a device matrix.
template<Format F>
__global__ void gemv(const unsigned char* weights, const float* scales, const float* x, float* y,
                     size_t rows, size_t cols, size_t scale_cols, size_t row_phase, bool accumulate) {
    const unsigned lane = threadIdx.x % warp;
    for (size_t r = size_t(blockIdx.x) * (threads / warp) + threadIdx.x / warp;
         r < rows; r += size_t(gridDim.x) * (threads / warp)) {
        float sum = 0.0f;
        for (size_t c = lane; c < cols; c += warp) {
            const size_t i = r * cols + c;
            float w;
            if constexpr (F == Format::fp8) {
                w = __uint_as_float(detail::e4m3fn_f32_bits(weights[i]));
                w *= scales[((r + row_phase) / block_side) * scale_cols + c / block_side];
            } else if constexpr (F == Format::bf16) {
                w = kernels::f32_from_bf16(reinterpret_cast<const uint16_t*>(weights)[i]);
            } else if constexpr (F == Format::fp16) {
                w = __half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(weights)[i]));
            } else {
                w = reinterpret_cast<const float*>(weights)[i];
            }
            sum = fmaf(w, x[c], sum);
        }
        for (unsigned offset = warp / 2; offset; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) y[r] = accumulate ? y[r] + sum : sum;
    }
}

struct FreeDevice { void operator()(unsigned char* p) const noexcept { cudaFree(p); } };
struct FreeHost   { void operator()(unsigned char* p) const noexcept { cudaFreeHost(p); } };
struct FreeStream { void operator()(cudaStream_t s) const noexcept { cudaStreamDestroy(s); } };
struct FreeEvent  { void operator()(cudaEvent_t e) const noexcept { cudaEventDestroy(e); } };
using Stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, FreeStream>;
using Event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, FreeEvent>;
using Pinned = std::unique_ptr<unsigned char, FreeHost>;

Stream new_stream() {
    cudaStream_t s;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "create stream");
    return Stream(s);
}
Event new_event() {
    cudaEvent_t e;
    check(cudaEventCreate(&e), "create timing event");
    return Event(e);
}
Pinned new_stage() {
    unsigned char* p;
    check(cudaMallocHost(&p, stage_bytes), "allocate pinned staging buffer");
    return Pinned(p);
}
float elapsed_ms(cudaEvent_t begin, cudaEvent_t end) {
    float elapsed;
    check(cudaEventElapsedTime(&elapsed, begin, end), "measure operation");
    return elapsed;
}

class Buffer {
    std::unique_ptr<unsigned char, FreeDevice> ptr_;
    size_t size_ = 0;
public:
    unsigned char* get() const { return ptr_.get(); }
    size_t size() const { return size_; }
    void reserve(size_t bytes) {
        if (bytes <= size_) return;
        ptr_.reset();  // do not hold both the old and the new allocation
        size_ = 0;
        unsigned char* p;
        check(cudaMalloc(&p, bytes), "allocate device buffer");
        ptr_.reset(p);
        size_ = bytes;
    }
};

struct Stats {
    uint64_t hits = 0, misses = 0, expert_hits = 0, expert_misses = 0, evictions = 0;
    uint64_t streamed = 0, launches = 0, h2d = 0, d2h = 0, weight_upload = 0;
    uint64_t overlap_calls = 0, overlap_tiles = 0;
    size_t resident = 0, peak = 0;
    double kernel_ms = 0, h2d_ms = 0, d2h_ms = 0;
};

struct Entry {
    std::string key;
    Format format;
    size_t rows, cols;
    Buffer data;
};

// The existing pinned stage, device scratch and compute stream serve slot 0.
// Allocate the second slot and its copy stream only on the first eligible call.
struct Pipeline {
    struct Flight {
        Event copy_begin = new_event(), copied = new_event();
        Event kernel_begin = new_event(), done = new_event();
        size_t pending_bytes = 0;
    };
    Stream copy = new_stream();
    Pinned extra_stage = new_stage();
    Buffer extra_scratch;
    Flight flight[2];
};

class State {
    const size_t budget_;
    const bool overlap_;
    Stream stream_ = new_stream();
    Event begin_ = new_event(), end_ = new_event();
    Pinned stage_ = new_stage();
    Buffer x_, y_, scratch_;
    std::unique_ptr<Pipeline> pipe_;
    std::list<Entry> lru_;    // MRU first; iterators stable across splice
    std::unordered_map<std::string, std::list<Entry>::iterator> by_key_;
    Stats st_;

    void start() { check(cudaEventRecord(begin_.get(), stream_.get()), "record start"); }
    void finish(double& milliseconds) {
        check(cudaEventRecord(end_.get(), stream_.get()), "record end");
        check(cudaEventSynchronize(end_.get()), "wait for operation");
        milliseconds += elapsed_ms(begin_.get(), end_.get());
    }

    // Serial transfers use slot 0. The stream event completes before reuse.
    void transfer(void* device, void* host_out, const void* host_in, size_t bytes) {
        auto* d = static_cast<unsigned char*>(device);
        auto* out = static_cast<unsigned char*>(host_out);
        auto* in = static_cast<const unsigned char*>(host_in);
        while (bytes) {
            const size_t part = std::min(bytes, stage_bytes);
            if (in) {
                std::memcpy(stage_.get(), in, part);
                start();
                check(cudaMemcpyAsync(d, stage_.get(), part, cudaMemcpyHostToDevice, stream_.get()), "upload");
                finish(st_.h2d_ms);
                st_.h2d += part;
                in += part;
            } else {
                start();
                check(cudaMemcpyAsync(stage_.get(), d, part, cudaMemcpyDeviceToHost, stream_.get()), "download");
                finish(st_.d2h_ms);
                std::memcpy(out, stage_.get(), part);
                st_.d2h += part;
                out += part;
            }
            d += part;
            bytes -= part;
        }
    }
    void upload(Buffer& b, Layout layout, const unsigned char* raw, const float* scales) {
        b.reserve(layout.total);
        transfer(b.get(), nullptr, raw, layout.raw);
        if (layout.scale_bytes)
            transfer(b.get() + layout.scale_offset, nullptr, scales, layout.scale_bytes);
        st_.weight_upload += layout.raw + layout.scale_bytes;
    }

    void enqueue_gemv(Format f, const Buffer& b, Layout layout, size_t row0, size_t col0,
                      size_t rows, size_t cols) {
        const auto* scales = reinterpret_cast<const float*>(b.get() + layout.scale_offset);
        const auto* x = reinterpret_cast<const float*>(x_.get()) + col0;
        auto* y = reinterpret_cast<float*>(y_.get()) + row0;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, div_up(rows, threads / warp)));
        const size_t sc = div_up(cols, block_side), phase = row0 % block_side;
        #define LAUNCH(F) gemv<F><<<blocks, threads, 0, stream_.get()>>>( \
            b.get(), scales, x, y, rows, cols, sc, phase, col0 != 0)
        switch (f) {
        case Format::fp8:  LAUNCH(Format::fp8);  break;
        case Format::bf16: LAUNCH(Format::bf16); break;
        case Format::fp16: LAUNCH(Format::fp16); break;
        case Format::fp32: LAUNCH(Format::fp32); break;
        }
        #undef LAUNCH
        check(cudaGetLastError(), "launch gemv");
    }

    void launch(Format f, const Buffer& b, Layout layout, size_t row0, size_t col0,
                size_t rows, size_t cols) {
        start();
        enqueue_gemv(f, b, layout, row0, col0, rows, cols);
        finish(st_.kernel_ms);
        ++st_.launches;
    }

    // Waiting on this slot's last kernel also fences both H2D copies, via
    // cudaStreamWaitEvent below. Neither buffer nor its timing events can be
    // reused until the counters have consumed that generation of the events.
    void retire(Pipeline::Flight& slot) {
        if (!slot.pending_bytes) return;
        check(cudaEventSynchronize(slot.done.get()), "wait for streamed kernel");
        const float copy_ms = elapsed_ms(slot.copy_begin.get(), slot.copied.get());
        const float compute_ms = elapsed_ms(slot.kernel_begin.get(), slot.done.get());
        st_.h2d_ms += copy_ms;
        st_.kernel_ms += compute_ms;
        st_.h2d += slot.pending_bytes;
        st_.weight_upload += slot.pending_bytes;
        ++st_.launches;
        ++st_.streamed;
        ++st_.overlap_tiles;
        slot.pending_bytes = 0;
    }

    bool pipeline_rows(Format f, const unsigned char* raw, const Vec& scales, size_t rows, size_t cols) {
        const size_t row_bytes = cols * width(f);  // already validated by describe()
        size_t height = stage_bytes / row_bytes;
        if (!height || rows <= height) return false; // wide split row or a single raw tile
        const size_t sc = div_up(cols, block_side);
        if (f == Format::fp8) {
            // Raw and scales share one 1 MiB pinned slot. Allow for a tile
            // starting anywhere inside a row block, plus alignment padding.
            const size_t overhead = 3 + sizeof(float) * sc * div_up(block_side - 1 + height, block_side);
            if (overhead >= stage_bytes) return false;
            height = (stage_bytes - overhead) / row_bytes;
            if (!height) return false; // even a whole packed row cannot fit
        }
        if (!pipe_) pipe_ = std::make_unique<Pipeline>();
        // No CUDA allocation (an implicit barrier) inside the active pipeline.
        scratch_.reserve(stage_bytes);
        pipe_->extra_scratch.reserve(stage_bytes);
        unsigned next = 0;
        for (size_t r = 0; r < rows; next ^= 1) {
            auto& slot = pipe_->flight[next];
            retire(slot);
            const size_t nr = std::min(rows - r, height);
            const size_t ns = f == Format::fp8 ? div_up(r % block_side + nr, block_side) * sc : 0;
            const Layout tile(nr * row_bytes, ns);
            unsigned char* host = next ? pipe_->extra_stage.get() : stage_.get();
            Buffer& device = next ? pipe_->extra_scratch : scratch_;
            std::memcpy(host, raw + r * row_bytes, tile.raw);
            if (ns) std::memcpy(host + tile.scale_offset, scales.data() + r / block_side * sc, tile.scale_bytes);

            const cudaStream_t copy = pipe_->copy.get();
            check(cudaEventRecord(slot.copy_begin.get(), copy), "time streamed upload");
            check(cudaMemcpyAsync(device.get(), host, tile.raw, cudaMemcpyHostToDevice, copy), "upload streamed rows");
            if (ns) check(cudaMemcpyAsync(device.get() + tile.scale_offset, host + tile.scale_offset,
                                          tile.scale_bytes, cudaMemcpyHostToDevice, copy), "upload streamed scales");
            check(cudaEventRecord(slot.copied.get(), copy), "mark streamed upload complete");
            check(cudaStreamWaitEvent(stream_.get(), slot.copied.get(), 0), "order streamed gemv after upload");
            check(cudaEventRecord(slot.kernel_begin.get(), stream_.get()), "time streamed kernel");
            enqueue_gemv(f, device, tile, r, 0, nr, cols);
            check(cudaEventRecord(slot.done.get(), stream_.get()), "mark streamed kernel complete");
            slot.pending_bytes = tile.raw + tile.scale_bytes;
            r += nr;
        }
        for (auto& slot : pipe_->flight) retire(slot); // also the short tail, before D2H reuses slot 0
        ++st_.overlap_calls;
        return true;
    }

    void stream_tiles(Format f, const unsigned char* raw, const Vec& scales, size_t rows, size_t cols) {
        if (overlap_ && pipeline_rows(f, raw, scales, rows, cols)) return;
        // Prefer whole rows so both file and scale slices are contiguous. If a
        // single row is larger than the cap, split it at 128-column boundaries.
        const size_t w = width(f), tile_cols = std::min(cols, stage_bytes / w);
        const size_t max_rows = cols == tile_cols ? stage_bytes / (cols * w) : 1;
        const size_t full_sc = div_up(cols, block_side);
        for (size_t r = 0; r < rows;) {
            const size_t nr = std::min(rows - r, max_rows);
            for (size_t c = 0; c < cols;) {
                const size_t nc = std::min(cols - c, tile_cols);
                const size_t ns = f == Format::fp8 ?
                    div_up((r % block_side) + nr, block_side) * div_up(nc, block_side) : 0;
                const float* sc = ns ? scales.data() + (r / block_side) * full_sc + c / block_side : nullptr;
                const Layout tile(nr * nc * w, ns);
                upload(scratch_, tile, raw + (r * cols + c) * w, sc);
                launch(f, scratch_, tile, r, c, nr, nc);
                ++st_.streamed;
                c += nc;
            }
            r += nr;
        }
    }

    Vec begin_product(const Vec& x, size_t rows) {
        const size_t input_bytes = mul(x.size(), sizeof(float)), output_bytes = mul(rows, sizeof(float));
        Vec result(rows);
        x_.reserve(input_bytes);
        y_.reserve(output_bytes);
        transfer(x_.get(), nullptr, x.data(), input_bytes);
        return result;
    }
    void download_result(Vec& out) {
        transfer(y_.get(), out.data(), nullptr, out.size() * sizeof(float));
    }

    // Shared by a loader probe (before file I/O) and a direct multiply. A
    // negative probe has no side effects and is not a completed cache miss.
    bool use_resident(const std::string& key, Format f, Layout layout, size_t rows, size_t cols,
                      const Vec& x, Vec& result) {
        const auto found = by_key_.find(key);
        if (found == by_key_.end()) return false;
        const auto& entry = *found->second;
        if (entry.format != f || entry.rows != rows || entry.cols != cols)
            throw std::invalid_argument("GigaChat4 CUDA: cache key reused for a different tensor shape/dtype: " + key);
        Vec out = begin_product(x, rows);
        lru_.splice(lru_.begin(), lru_, found->second);
        launch(f, entry.data, layout, 0, 0, rows, cols);
        download_result(out);
        ++st_.hits;
        st_.expert_hits += key.find(".experts.") != std::string::npos;
        result = std::move(out);
        return true;
    }

public:
    State(size_t budget, bool overlap) : budget_(budget), overlap_(overlap) {}
    void drain() noexcept {
        // On an enqueue/record error work may be in flight without a valid
        // completion event. Drain both streams before any buffer can be reused.
        if (pipe_) cudaStreamSynchronize(pipe_->copy.get());
        cudaStreamSynchronize(stream_.get());
        if (pipe_) for (auto& slot : pipe_->flight) slot.pending_bytes = 0;
    }
    ~State() { drain(); }

    bool multiply_cached(const std::string& key, const std::string& dtype, size_t rows,
                         size_t cols, const Vec& x, Vec& result) {
        const Format f = parse(dtype);
        return use_resident(key, f, describe(f, rows, cols, x), rows, cols, x, result);
    }

    Vec multiply(const std::string& key, const unsigned char* raw, size_t nbytes, const std::string& dtype,
                 const Vec& scales, size_t rows, size_t cols, const Vec& x) {
        const Format f = parse(dtype);
        const Layout layout = describe(f, rows, cols, x);
        if (!raw || nbytes != layout.raw)
            throw std::invalid_argument("GigaChat4 CUDA: missing or wrong-sized weight bytes for " + key);
        if (scales.size() != layout.scale_bytes / sizeof(float))
            throw std::invalid_argument("GigaChat4 CUDA: unexpected scale count for " + key);
        Vec result;
        if (use_resident(key, f, layout, rows, cols, x, result)) return result;
        result = begin_product(x, rows);
        if (layout.total > budget_) {
            stream_tiles(f, raw, scales, rows, cols);
        } else {
            while (st_.resident > budget_ - layout.total) {
                const auto& old = lru_.back();
                st_.resident -= old.data.size();
                by_key_.erase(old.key);
                lru_.pop_back();
                ++st_.evictions;
            }
            Entry fresh{key, f, rows, cols, {}};
            upload(fresh.data, layout, raw, scales.data());
            lru_.push_front(std::move(fresh));
            try {
                by_key_.emplace(key, lru_.begin());
            } catch (...) {
                lru_.pop_front();
                throw;
            }
            st_.resident += layout.total;
            st_.peak = std::max(st_.peak, st_.resident);
            launch(f, lru_.front().data, layout, 0, 0, rows, cols);
        }
        download_result(result);
        ++st_.misses;
        st_.expert_misses += key.find(".experts.") != std::string::npos;
        return result;
    }

    Config metrics() const {
        return {{"cache_budget_bytes", budget_}, {"cache_bytes", st_.resident}, {"peak_cache_bytes", st_.peak},
                {"cache_entries", by_key_.size()}, {"cache_hits", st_.hits}, {"cache_misses", st_.misses},
                {"expert_cache_hits", st_.expert_hits}, {"expert_cache_misses", st_.expert_misses},
                {"cache_evictions", st_.evictions}, {"streamed_tiles", st_.streamed}, {"kernel_launches", st_.launches},
                {"overlap_enabled", overlap_}, {"overlap_calls", st_.overlap_calls}, {"overlap_tiles", st_.overlap_tiles},
                {"pinned_bytes", stage_bytes * (pipe_ ? 2 : 1)},
                {"stream_scratch_bytes", scratch_.size() + (pipe_ ? pipe_->extra_scratch.size() : 0)},
                {"input_buffer_bytes", x_.size()}, {"output_buffer_bytes", y_.size()},
                {"h2d_bytes", st_.h2d}, {"d2h_bytes", st_.d2h}, {"weight_h2d_bytes", st_.weight_upload},
                {"kernel_ms", st_.kernel_ms}, {"h2d_ms", st_.h2d_ms}, {"d2h_ms", st_.d2h_ms}};
    }
};

// CUDA's active device is per host thread. Switch back even on an exception.
class OnDevice {
    int previous_;
    bool changed_;
public:
    explicit OnDevice(int target) {
        check(cudaGetDevice(&previous_), "get device");
        changed_ = previous_ != target;
        if (changed_) check(cudaSetDevice(target), "select matvec device");
    }
    ~OnDevice() { if (changed_) cudaSetDevice(previous_); }
};

class Backend final : public MatvecBackend {
    int device_;
    std::unique_ptr<State> state_;
    mutable std::mutex mutex_;
    template<class F> auto on_device(F work) {
        std::lock_guard<std::mutex> lock(mutex_);
        OnDevice active(device_);
        try {
            return work();
        } catch (...) {
            // An error recording a completion event may leave a DMA outstanding.
            // Drain it before the staging buffer is ever borrowed again.
            state_->drain();
            throw;
        }
    }
public:
    Backend(size_t budget, bool overlap) {
        check(cudaGetDevice(&device_), "get device");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device_), "get device properties");
        if (prop.major < 7) throw std::runtime_error("GigaChat4 CUDA requires compute capability 7.0 or later");
        state_ = std::make_unique<State>(budget, overlap);
    }
    ~Backend() override {
        // Destruct on the owning device, including when called from a different
        // host thread. Cleanup is best effort if the driver is shutting down.
        int prev = -1;
        cudaGetDevice(&prev);
        if (prev != device_) cudaSetDevice(device_);
        state_.reset();
        if (prev >= 0 && prev != device_) cudaSetDevice(prev);
    }
    Vec multiply(const std::string& key, const unsigned char* raw, size_t nbytes, const std::string& dtype,
                 const Vec& scales, size_t rows, size_t cols, const Vec& x) override {
        return on_device([&]{ return state_->multiply(key, raw, nbytes, dtype, scales, rows, cols, x); });
    }
    bool multiply_cached(const std::string& key, const std::string& dtype, size_t rows, size_t cols,
                         const Vec& x, Vec& result) override {
        return on_device([&]{ return state_->multiply_cached(key, dtype, rows, cols, x, result); });
    }
    Config metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        Config result = state_->metrics();
        result["device"] = device_;
        return result;
    }
};

} // namespace

std::unique_ptr<MatvecBackend> make_cuda_backend(size_t cache_bytes, bool overlap) {
    return std::make_unique<Backend>(cache_bytes, overlap);
}
Config cuda_metrics(const MatvecBackend& backend) {
    const auto* cuda = dynamic_cast<const Backend*>(&backend);
    if (!cuda) throw std::invalid_argument("cuda_metrics requires a GigaChat4 CUDA backend");
    return cuda->metrics();
}

} // namespace strata::gigachat4
