#include "weights.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::gigachat4 {
namespace {

namespace fs = std::filesystem;
using Bytes = std::vector<unsigned char>;
constexpr size_t MiB = 1024 * 1024;
constexpr size_t block = 128, tile_target = 16 * MiB, max_buffer = 64 * MiB;
constexpr size_t max_header = 32 * MiB, max_headers_total = 128 * MiB;
constexpr size_t max_tensors = 200000, max_shards = 256, max_entries = 65536;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("GigaChat4 weights: " + message);
}

uint64_t natural(const Config& j, const std::string& field) {
    if (!j.is_number_unsigned()) fail("expected unsigned integer in " + field);
    return j.get<uint64_t>();
}

uint64_t product(uint64_t a, uint64_t b, const std::string& field) {
    if (a && b > UINT64_MAX / a) fail("size overflow: " + field);
    return a * b;
}

size_t blocks(size_t n) { return n / block + (n % block != 0); }

struct CloseFile { void operator()(std::FILE* f) const noexcept { std::fclose(f); } };
using File = std::unique_ptr<std::FILE, CloseFile>;

File open_file(const fs::path& path) {
    if (!fs::is_regular_file(path)) fail("not a regular file: " + path.string());
#if defined(_WIN32)
    File f(_wfopen(path.c_str(), L"rb"));
#else
    File f(std::fopen(path.c_str(), "rb"));
#endif
    if (!f) fail("cannot open " + path.string());
    // Do not have stdio pull payload into a buffer when asking for the final
    // header bytes (or adjacent rows when reading a single embedding).
    if (std::setvbuf(f.get(), nullptr, _IONBF, 0) != 0) fail("cannot set exact reads on " + path.string());
    return f;
}

void read_at(const File& f, uint64_t off, void* dest, size_t n, const fs::path& path) {
#if defined(_WIN32)
    const bool seek_ok = off <= INT64_MAX && _fseeki64(f.get(), static_cast<int64_t>(off), SEEK_SET) == 0;
#else
    const bool seek_ok = off <= uint64_t(std::numeric_limits<off_t>::max()) &&
                         fseeko(f.get(), static_cast<off_t>(off), SEEK_SET) == 0;
#endif
    if (!seek_ok || std::fread(dest, 1, n, f.get()) != n)
        fail("short or failed read in " + path.string() + " at " + std::to_string(off));
}

// JSON's default last-key-wins behavior would hide conflicting offsets/index entries.
Config parse(const std::string& text, const fs::path& path) {
    if (text.find('\0') != std::string::npos) fail("NUL byte in JSON: " + path.string());
    using Event = Config::parse_event_t;
    std::vector<std::unordered_set<std::string>> keys;
    size_t events = 0;
    auto check = [&](int depth, Event event, Config& value) {
        if (depth > 32 || ++events > 4000000) fail("JSON too complex: " + path.string());
        if (event == Event::object_start) keys.emplace_back();
        if (event == Event::object_end) keys.pop_back();
        if (event == Event::key && !keys.back().insert(value.get_ref<const std::string&>()).second)
            fail("duplicate JSON key in " + path.string());
        return true;
    };
    try {
        auto j = Config::parse(text, check);
        if (!j.is_object()) fail("expected an object in " + path.string());
        return j;
    } catch (const Config::exception& e) {
        fail(path.string() + ": " + e.what());
    }
}

Config text_file(const fs::path& path, size_t limit) {
    auto f = open_file(path);
    auto len = fs::file_size(path);
    if (!len || len > limit) fail("invalid JSON length: " + path.string());
    std::string text(static_cast<size_t>(len), '\0');
    read_at(f, 0, text.data(), text.size(), path);
    return parse(text, path);
}

void check_config(const fs::path& path) {
    if (!fs::exists(path)) return;
    const auto cfg = text_file(path, MiB);
    const auto q = cfg.find("quantization_config");
    if (q == cfg.end() || q->is_null()) return;
    if (!q->is_object() || q->at("quant_method") != "fp8" ||
        q->at("fmt") != "e4m3" || q->at("weight_block_size") != Config::array({128, 128}))
        fail("expected fp8/e4m3 weight_block_size [128,128] in " + path.string());
}

enum class Type { f32, bf16, f16, fp8 };

Type dtype(const std::string& tag) {
    if (tag == "F32") return Type::f32;
    if (tag == "BF16") return Type::bf16;
    if (tag == "F16") return Type::f16;
    if (tag == "F8_E4M3" || tag == "F8_E4M3FN") return Type::fp8;
    fail("unsupported dtype " + tag);
}

const char* tag(Type t) {
    switch (t) {
        case Type::f32: return "F32";
        case Type::bf16: return "BF16";
        case Type::f16: return "F16";
        case Type::fp8: return "F8_E4M3";
    }
    fail("invalid dtype");
}

size_t width(Type t) { return t == Type::f32 ? 4 : (t == Type::fp8 ? 1 : 2); }

float float_bits(uint32_t bits) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "float32 required");
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
uint32_t le16(const unsigned char* p) { return p[0] | (uint32_t(p[1]) << 8); }
float f16(uint32_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 31, m = h & 1023;
    if (exp == 31) return float_bits(sign | 0x7f800000u | (m << 13));
    if (!exp) {
        if (!m) return float_bits(sign);
        exp = 113;
        while (!(m & 1024)) { m <<= 1; --exp; }
        m &= 1023;
    } else {
        exp += 112;
    }
    return float_bits(sign | (exp << 23) | (m << 13));
}

const std::array<float, 256>& fp8_table() {
    static const auto table = [] {
        std::array<float, 256> a{};
        for (unsigned i = 0; i < a.size(); ++i) {
            int exp = (i >> 3) & 15, m = i & 7;
            float v = exp ? std::ldexp(float(8 + m), exp - 10) : std::ldexp(float(m), -9);
            if ((i & 127) == 127) v = std::numeric_limits<float>::quiet_NaN();
            a[i] = i & 128 ? -v : v;
        }
        return a;
    }();
    return table;
}

// Select once per slice, not once per element in the CPU inner product.
template<class Fn>
void with_dense_decoder(Type type, Fn fn) {
    if (type == Type::f32)
        fn([](const unsigned char* p, size_t i) {
            p += i * 4;
            return float_bits(le16(p) | (le16(p + 2) << 16));
        });
    else if (type == Type::bf16)
        fn([](const unsigned char* p, size_t i) { return float_bits(le16(p + i * 2) << 16); });
    else if (type == Type::f16)
        fn([](const unsigned char* p, size_t i) { return f16(le16(p + i * 2)); });
    else fail("expected dense dtype");
}

struct Tensor {
    size_t shard;
    uint64_t start, bytes;
    Type type;
    std::vector<size_t> shape;
    const Tensor* scale = nullptr; // linked once the complete index is available
};

void check_name(const std::string& name) {
    if (name.empty() || name.size() > 512 || name.find('\0') != std::string::npos)
        fail("invalid tensor name");
}

void check_vec(uint64_t n, const std::string& name) {
    if (!n || n > max_buffer / sizeof(float)) fail("vector size out of bounds: " + name);
}

void cpu_multiply(float* out, const unsigned char* raw, Type type, const Vec& scales,
                  size_t rows, size_t cols, const Vec& x) {
    if (type == Type::fp8) {
        const auto& lut = fp8_table();
        const size_t stride = blocks(cols);
        for (size_t r = 0; r < rows; ++r) {
            float sum = 0;
            for (size_t j = 0; j < cols; j += block) {
                const float scale = scales[(r / block) * stride + j / block];
                for (size_t c = j; c < std::min(j + block, cols); ++c)
                    sum += (lut[raw[r * cols + c]] * scale) * x[c];
            }
            out[r] = sum;
        }
    } else {
        with_dense_decoder(type, [&](auto decode) {
            for (size_t r = 0; r < rows; ++r) {
                float sum = 0;
                for (size_t c = 0; c < cols; ++c) sum += decode(raw, r * cols + c) * x[c];
                out[r] = sum;
            }
        });
    }
}

} // namespace

struct FileWeights::Impl {
    struct Shard { fs::path path; File file; };
    using Range = std::tuple<size_t, uint64_t, size_t>; // shard, absolute offset, length
    using Buffer = std::shared_ptr<const Bytes>;
    using Lru = std::list<std::pair<Range, Buffer>>;
    std::vector<Shard> shards;
    std::unordered_map<std::string, Tensor> tensors;
    Lru lru;
    std::map<Range, Lru::iterator> cache;
    const size_t budget;
    MatvecBackend* backend;
    std::string key_prefix;
    WeightStats stats;
    uint64_t header_bytes = 0, payload_bytes = 0;

    Impl(const fs::path& dir, size_t budget_, MatvecBackend* backend_)
        : budget(budget_), backend(backend_) {
        static std::atomic<uint64_t> next_id{0};
        key_prefix = "giga4:" + std::to_string(++next_id) + ":";
        check_config(dir / "config.json");
        const auto idx = dir / "model.safetensors.index.json";
        if (fs::exists(idx)) {
            const auto index = text_file(idx, max_header);
            const auto& map = index.at("weight_map");
            if (!map.is_object() || map.empty() || map.size() > max_tensors) fail("invalid weight_map");
            std::set<std::string> files;
            for (const auto& item : map.items()) {
                check_name(item.key());
                const auto& filename = item.value().get_ref<const std::string&>();
                if (filename.size() > 255 || filename.find_first_of("/\\:") != std::string::npos ||
                    filename.find('\0') != std::string::npos || fs::path(filename).extension() != ".safetensors")
                    fail("invalid shard filename: " + filename);
                files.insert(filename);
                if (files.size() > max_shards) fail("too many shards");
            }
            tensors.reserve(map.size());
            for (const auto& filename : files) add_shard(dir / filename, &map);
            if (map.size() != tensors.size()) fail("index entry missing from shard header(s)");
            if (index.contains("metadata") && index.at("metadata").contains("total_size") &&
                natural(index.at("metadata").at("total_size"), "total_size") != payload_bytes)
                fail("index total_size differs from payload bytes");
        } else {
            add_shard(dir / "model.safetensors", nullptr);
        }
        if (tensors.empty()) fail("empty checkpoint");
        constexpr char suffix[] = "_scale_inv";
        for (auto& kv : tensors) {
            const auto& name = kv.first;
            auto& t = kv.second;
            if (t.type == Type::fp8) {
                auto s = tensors.find(name + suffix);
                if (t.shape.size() != 2 || s == tensors.end() || s->second.type == Type::fp8 ||
                    s->second.shape != std::vector<size_t>{blocks(t.shape[0]), blocks(t.shape[1])})
                    fail("missing or invalid 128x128 scale grid: " + name);
                t.scale = &s->second;
            } else if (name.size() >= sizeof(suffix)-1 &&
                       name.compare(name.size()-(sizeof(suffix)-1), sizeof(suffix)-1, suffix) == 0) {
                const auto src = tensors.find(name.substr(0, name.size()-(sizeof(suffix)-1)));
                if (src == tensors.end() || src->second.type != Type::fp8)
                    fail("scale grid without an FP8 weight: " + name);
            }
        }
        stats.shards = shards.size();
        stats.tensors = tensors.size();
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
        stats.prefetch_supported = true;
#endif
    }

    void add_shard(const fs::path& path, const Config* index) {
        auto file = open_file(path);
        const uint64_t size = fs::file_size(path);
        unsigned char len[8];
        read_at(file, 0, len, 8, path);
        uint64_t h = 0;
        for (int i = 7; i >= 0; --i) h = (h << 8) | len[i];
        if (!h || h > max_header || h > size - 8 || h > max_headers_total - header_bytes)
            fail("invalid header length in " + path.string());
        header_bytes += h;
        std::string text(static_cast<size_t>(h), '\0');
        read_at(file, 8, text.data(), text.size(), path);
        if (text.front() != '{') fail("invalid safetensors header in " + path.string());
        const auto header = parse(text, path);
        const uint64_t base = h + 8, payload = size - base;
        if (payload > UINT64_MAX - payload_bytes) fail("checkpoint size overflow");
        payload_bytes += payload;
        std::vector<std::pair<uint64_t, uint64_t>> spans;
        const std::string filename = path.filename().string();
        for (const auto& item : header.items()) {
            const auto& name = item.key();
            const auto& j = item.value();
            if (name == "__metadata__") {
                if (!j.is_object() || !std::all_of(j.begin(), j.end(), [](const auto& v) { return v.is_string(); }))
                    fail("invalid safetensors metadata in " + filename);
                continue;
            }
            check_name(name);
            if (index && (!index->contains(name) || index->at(name) != filename))
                fail("shard disagrees with index for " + name);
            const auto& shape = j.at("shape");
            const auto& offsets = j.at("data_offsets");
            Tensor t{shards.size(), 0, 0, dtype(j.at("dtype").get<std::string>()), {}};
            if (!shape.is_array() || shape.size() > 8 || !offsets.is_array() || offsets.size() != 2)
                fail("invalid shape/offsets: " + name);
            uint64_t bytes = width(t.type);
            for (const auto& v : shape) {
                const auto d = natural(v, name);
                if (!d || d > SIZE_MAX) fail("invalid dimension: " + name);
                t.shape.push_back(static_cast<size_t>(d));
                bytes = product(bytes, d, name);
            }
            const uint64_t b = natural(offsets[0], name), e = natural(offsets[1], name);
            if (b > e || e > payload || e - b != bytes) fail("bad byte extent: " + name);
            t.start = base + b;
            t.bytes = bytes;
            if (tensors.size() >= max_tensors || !tensors.emplace(name, std::move(t)).second)
                fail("duplicate or excessive tensors: " + name);
            spans.emplace_back(b, e);
        }
        std::sort(spans.begin(), spans.end());
        uint64_t end = 0;
        for (const auto& span : spans) {
            if (span.first != end) fail("gap or overlap in " + filename);
            end = span.second;
        }
        if (end != payload) fail("trailing payload in " + filename);
        shards.push_back({path, std::move(file)});
    }

    const Tensor& find(const std::string& name, const std::vector<size_t>& shape) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) fail("missing tensor: " + name);
        if (it->second.shape != shape) fail("unexpected shape: " + name);
        return it->second;
    }

    Buffer slice(const Tensor& t, uint64_t off, size_t n) {
        if (!n || n > max_buffer || off > t.bytes || n > t.bytes - off) fail("invalid tensor slice");
        const Range key{t.shard, t.start + off, n};
        const bool cacheable = n <= budget;
        if (cacheable) {
            auto hit = cache.find(key);
            if (hit != cache.end()) {
                ++stats.cache_hits;
                lru.splice(lru.begin(), lru, hit->second);
                return hit->second->second;
            }
            while (stats.cache_bytes > budget - n || lru.size() >= max_entries) {
                const auto& victim = lru.back();
                stats.cache_bytes -= victim.second->size();
                cache.erase(victim.first);
                lru.pop_back();
                ++stats.evictions;
            }
        }
        ++stats.cache_misses;
        auto data = std::make_shared<Bytes>(n);
        auto& sh = shards[t.shard];
        const auto start = std::chrono::steady_clock::now();
        read_at(sh.file, t.start + off, data->data(), n, sh.path);
        stats.payload_read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        stats.bytes_read += n;
        ++stats.read_ops;
        stats.max_read_bytes = std::max(stats.max_read_bytes, n);
        if (cacheable) {
            lru.emplace_front(key, data);
            cache.emplace(key, lru.begin());
            stats.cache_bytes += n;
            stats.peak_cache_bytes = std::max(stats.peak_cache_bytes, stats.cache_bytes);
        }
        return data;
    }

    Vec dense_values(const Tensor& t, uint64_t first, size_t count) {
        check_vec(count, "dense read");
        auto raw = slice(t, first * width(t.type), count * width(t.type));
        Vec result(count);
        with_dense_decoder(t.type, [&](auto decode) {
            for (size_t i = 0; i < count; ++i) result[i] = decode(raw->data(), i);
        });
        return result;
    }

    // Works for row() starting inside a block as well as aligned linear() tiles.
    Vec scales(const Tensor& t, size_t first, size_t rows) {
        if (!t.scale) return {};
        size_t stride = t.scale->shape[1], start = first / block;
        Vec values = dense_values(*t.scale, uint64_t(start) * stride, (blocks(first + rows) - start) * stride);
        if (!std::all_of(values.begin(), values.end(), [](float s) { return std::isfinite(s); }))
            fail("non-finite FP8 block scale");
        return values;
    }

    Vec fp8_values(const Tensor& t, size_t first, size_t rows) {
        size_t cols = t.shape[1];
        check_vec(uint64_t(rows) * cols, "FP8 read");
        auto s = scales(t, first, rows);
        auto raw = slice(t, uint64_t(first) * cols, rows * cols);
        Vec result(rows * cols);
        const auto& lut = fp8_table();
        for (size_t r = 0; r < rows; ++r) {
            size_t base = ((first + r) / block - first / block) * blocks(cols);
            for (size_t c = 0; c < cols; ++c)
                result[r * cols + c] = lut[(*raw)[r * cols + c]] * s[base + c / block];
        }
        return result;
    }

    void prefetch_expert(const std::string& prefix) {
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
        // Reserve room for all three projections; scales are normally tiny.
        // This path neither creates payload buffers nor changes host cache recency.
        std::array<const Tensor*, 3> ws{};
        size_t count = 0;
        for (const char* leaf : {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}) {
            const auto it = tensors.find(prefix + leaf);
            if (it != tensors.end()) ws[count++] = &it->second;
        }
        size_t remaining = max_buffer;
        auto advise = [&](const Tensor& t, size_t limit) {
            const size_t len = static_cast<size_t>(std::min<uint64_t>(t.bytes, limit));
            if (!len) return; // POSIX interprets length zero as "to end of file".
            remaining -= len;
            ++stats.prefetch_requests;
            stats.prefetch_bytes += len;
            if (t.start > uint64_t(std::numeric_limits<off_t>::max()) ||
                posix_fadvise(fileno(shards[t.shard].file.get()), static_cast<off_t>(t.start),
                              static_cast<off_t>(len), POSIX_FADV_WILLNEED) != 0)
                ++stats.prefetch_errors;
        };
        for (size_t i = 0; i < count; ++i)
            if (ws[i]->scale) advise(*ws[i]->scale, max_buffer / (2 * count));
        for (size_t i = 0; i < count; ++i)
            advise(*ws[i], remaining / (count - i));
#else
        (void)prefix;
#endif
    }
};

FileWeights::FileWeights(const fs::path& dir, size_t cache_bytes, MatvecBackend* backend)
    : impl_(std::make_unique<Impl>(dir, cache_bytes, backend)) {}
FileWeights::~FileWeights() = default;
WeightStats FileWeights::stats() const { return impl_->stats; }
void FileWeights::prefetch_expert(const std::string& prefix) { impl_->prefetch_expert(prefix); }

Vec FileWeights::read(const std::string& name, const std::vector<size_t>& shape) {
    const auto& t = impl_->find(name, shape);
    auto n = t.bytes / width(t.type);
    check_vec(n, name);
    if (t.type == Type::fp8) return impl_->fp8_values(t, 0, shape[0]);
    return impl_->dense_values(t, 0, static_cast<size_t>(n));
}

Vec FileWeights::row(const std::string& name, size_t rows, size_t cols, size_t index) {
    const auto& t = impl_->find(name, {rows, cols});
    if (index >= rows) fail("row index out of bounds: " + name);
    check_vec(cols, name);
    if (t.type == Type::fp8) return impl_->fp8_values(t, index, 1);
    return impl_->dense_values(t, uint64_t(index) * cols, cols);
}

Vec FileWeights::linear(const std::string& name, const Vec& x, size_t out) {
    const size_t cols = x.size();
    const auto& t = impl_->find(name, {out, cols});
    check_vec(out, name);
    check_vec(cols, name);
    const size_t row_bytes = cols * width(t.type);
    const size_t hard_rows = max_buffer / row_bytes;
    size_t step = std::max(tile_target / row_bytes, std::min(out, block));
    if (step > hard_rows) fail("128-row tile exceeds 64 MiB: " + name);
    if (step < out) step = step / block * block;
    Vec y(out);
    for (size_t first = 0; first < out; first += step) {
        size_t rows = std::min(out - first, step);
        std::string key;
        Vec result;
        bool hit = false;
        if (impl_->backend) {
            key = impl_->key_prefix + name + ":rows=" + std::to_string(first) + ":" + std::to_string(first + rows);
            hit = impl_->backend->multiply_cached(key, tag(t.type), rows, cols, x, result);
        }
        if (!hit) {
            auto scales = impl_->scales(t, first, rows);
            auto raw = impl_->slice(t, uint64_t(first) * row_bytes, rows * row_bytes);
            if (impl_->backend)
                result = impl_->backend->multiply(key, raw->data(), raw->size(), tag(t.type), scales, rows, cols, x);
            else
                cpu_multiply(y.data() + first, raw->data(), t.type, scales, rows, cols, x);
        }
        if (impl_->backend) {
            if (result.size() != rows) fail("backend returned wrong tile height: " + name);
            std::copy(result.begin(), result.end(), y.begin() + first);
        }
    }
    return y;
}

} // namespace strata::gigachat4
