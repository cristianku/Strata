// Standalone offline reader for one routed projection in the official 708B Q4_K_M.
// Build and file formats: docs/gigachat4/GGUF.md. No engine or installer dependency.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/quant/dequant.hpp"
#include "hash/hash.h"

#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>

namespace {
namespace fs = std::filesystem;
using strata::quant::Type;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct Pin { uint64_t size; size_t header; const char* sha; };
// SHA-256 over exactly [0, end of tensor directory), excluding alignment padding.
// This both identifies the revision and bounds metadata before the general mmap reader opens it.
constexpr Pin pins[] = {
    {48721510688, 6122781, "cb76f73013545066d3d5a0c3a00554fb15b3806ee39c3e77ba17aeaf6feca016"},
    {48483994976,   14663, "ac8f3152e580269b9b0bc8190c3017c476f367d2eca5f0eb8b97040716c7f7e8"},
    {49667756416,   15223, "31c5109d5dccdb841f9175b533db74ffeac0e078c08c1db5c55729403f862f81"},
    {49769253568,   16040, "7ad627ed47e9222e80696eead3640a79b3af48dc723a0eaaa57c56fe1b8394f1"},
    {48492378752,   13925, "664f99bf835f7ae9e392669b96b15f7480e37140395353fe30d9b701500a8b27"},
    {49603235328,   14827, "54acd43f9d1eee9d60c55e8c9fde92825812fabfa654112f97cff57ffb724b28"},
    {48483994976,   14684, "91f0742ef072f158e8fa4c24b852099f89d9b4ca455432179266b478c0f1ffed"},
    {47420989280,   14675, "639d01d18fe682d901affe211fcf514e2da3803956b3bd75b2cb3c63250423f7"},
    {38801040352,   12768, "d6dd77cf825716eb61668c29a42e267700ed8033429718e8e0d1dcf6a11a616b"},
};

void verify_header(const fs::path& path) {
    for (size_t i = 0; i < std::size(pins); ++i) {
        const auto& p = pins[i];
        if (path.filename() != "GigaChat4-708B-A41B-Q4_K_M-0000" + std::to_string(i+1) + "-of-00009.gguf") continue;
        require(fs::is_regular_file(path) && fs::file_size(path) == p.size, "wrong shard size or file type");
        std::ifstream f(path, std::ios::binary);
        std::vector<unsigned char> header(p.header);
        require(bool(f.read(reinterpret_cast<char*>(header.data()), header.size())), "short GGUF header");
        require(hash_sha256_hex(header.data(), header.size()) == p.sha,
                "GGUF header does not match pinned revision 4b705bf");
        return;
    }
    throw std::runtime_error("expected an official Q4_K_M shard filename");
}

int number(const char* s, int lo, int hi, const char* what) {
    int v = 0;
    const auto end = s + std::strlen(s);
    const auto result = std::from_chars(s, end, v);
    require(result.ptr == end && result.ec == std::errc{} && v >= lo && v <= hi,
            std::string(what) + " must be in [" + std::to_string(lo) + "," + std::to_string(hi) + "]");
    return v;
}

void write_new(const char* path, const void* data, size_t n) {
    std::FILE* f = std::fopen(path, "wbx");
    require(f != nullptr, std::string("cannot create new output: ") + path);
    bool ok = std::fwrite(data, 1, n, f) == n;
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        std::remove(path);
        throw std::runtime_error("incomplete output");
    }
}

void decode(Type type, const uint8_t* src, float* dst, size_t n) {
    require(strata::quant::dequant_row(type, src, dst, n), "unsupported quant decoder");
    require(std::all_of(dst, dst+n, [](float v){return std::isfinite(v);}), "non-finite weight");
}
}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 5 ? argv[5] : "";
    if (!((mode == "pack" && argc == 7) || (mode == "block" && argc == 9) ||
          (mode == "matvec" && argc == 8))) {
        std::fprintf(stderr, "usage: gguf-expert SHARD LAYER EXPERT gate|up|down "
                             "{pack OUT | block ROW BLOCK OUT | matvec INPUT_F32 OUT_F32}\n");
        return 2;
    }
    try {
        const uint16_t le = 1;
        static_assert(sizeof(size_t) >= 8 && sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        require(*reinterpret_cast<const uint8_t*>(&le) == 1, "little-endian host required");
        const int layer = number(argv[2], 3, 63, "layer"), expert = number(argv[3], 0, 255, "expert");
        const std::string proj = argv[4];
        require(proj == "gate" || proj == "up" || proj == "down", "expected gate, up or down");
        const uint64_t cols = proj == "down" ? 2048 : 7168, rows = proj == "down" ? 7168 : 2048;
        verify_header(argv[1]);
        const strata::GgufFile shard(argv[1]);
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + proj + "_exps.weight";
        const auto* t = shard.find(name);
        require(t != nullptr, name + " is not in this shard (see GGUF.md for split boundaries)");
        require(t->shape == std::vector<uint64_t>{cols,rows,256}, "unexpected expert stack shape");
        const auto type = static_cast<Type>(t->type);
        require(type == Type::Q4_K || (proj == "down" && type == Type::Q6_K), "unexpected expert type");
        const auto row_bytes = strata::quant::row_bytes(t->type, cols);
        const uint64_t stride = rows * row_bytes, n = 256 * stride;
        const auto payload = shard.file_size() - shard.data_start();
        require(row_bytes && strata::tensor_payload_bytes(*t) == n && t->offset <= payload &&
                n <= payload - t->offset && t->offset % 32 == 0, "invalid stack byte range");
        const uint8_t* src = shard.tensor_data(*t) + expert * stride;
        if (mode == "pack") {
            write_new(argv[6], src, stride);
        } else if (mode == "block") {
            const int r = number(argv[6], 0, rows-1, "row"), b = number(argv[7], 0, cols/256-1, "block");
            std::array<float, 256> out;
            decode(type, src + r * row_bytes + b * strata::quant::row_bytes(t->type, 256), out.data(), out.size());
            write_new(argv[8], out.data(), sizeof(out));
        } else {
            require(fs::is_regular_file(argv[6]) && fs::file_size(argv[6]) == cols * sizeof(float),
                    "input must contain exactly one little-endian FP32 vector");
            std::vector<float> x(cols), tmp(cols), y(rows);
            std::ifstream in(argv[6], std::ios::binary);
            require(bool(in.read(reinterpret_cast<char*>(x.data()), cols * sizeof(float))), "short input");
            require(std::all_of(x.begin(), x.end(), [](float v){return std::isfinite(v);}), "non-finite input");
            for (size_t r = 0; r < rows; ++r) {
                decode(type, src + r * row_bytes, tmp.data(), cols);
                y[r] = std::inner_product(tmp.begin(), tmp.end(), x.begin(), 0.f);
                require(std::isfinite(y[r]), "non-finite projection output");
            }
            write_new(argv[7], y.data(), y.size()*sizeof(float));
        }
        std::fprintf(stderr, "model.layers.%d.mlp.experts.%d.%s_proj.weight: %s [%llu,%llu], "
                             "row_bytes=%zu expert_offset=%llu mode=%s\n", layer, expert, proj.c_str(),
                     t->type_name(), (unsigned long long)rows, (unsigned long long)cols, row_bytes,
                     (unsigned long long)(shard.data_start() + t->offset + expert*stride), mode.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gguf-expert: %s\n", e.what());
        return 1;
    }
}
