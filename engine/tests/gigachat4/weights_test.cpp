#include "weights.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace strata::gigachat4;
namespace fs = std::filesystem;
namespace {

using Bytes = std::vector<unsigned char>;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f, const char* why) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error(std::string("did not reject: ") + why);
}
struct Temp {
    fs::path dir;
    Temp() {
        for (int i = 0; i < 10; ++i) {
            dir = fs::temp_directory_path() / ("strata-weights-" + std::to_string(std::random_device{}()));
            if (fs::create_directory(dir)) return;
        }
        throw std::runtime_error("cannot make test directory");
    }
    ~Temp() { std::error_code ignored; fs::remove_all(dir, ignored); }
};

Bytes words(std::initializer_list<uint16_t> values) {
    Bytes b;
    for (uint16_t v : values) { b.push_back(v & 255); b.push_back(v >> 8); }
    return b;
}
Bytes floats(const Vec& values) {
    Bytes b;
    for (float v : values) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        for (int i = 0; i < 4; ++i) b.push_back((u >> (8*i)) & 255);
    }
    return b;
}
std::ofstream header(const fs::path& path, std::string json) {
    json.append((8 - json.size() % 8) % 8, ' ');
    std::ofstream f(path, std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    for (int i = 0; i < 8; ++i) f.put(static_cast<char>(uint64_t(json.size()) >> (8*i)));
    f.write(json.data(), static_cast<std::streamsize>(json.size()));
    return f;
}
struct Shard {
    Config h = Config::object();
    Bytes body;
    void add(const std::string& name, const char* dtype, const std::vector<size_t>& shape, const Bytes& raw) {
        h[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {body.size(), body.size() + raw.size()}}};
        body.insert(body.end(), raw.begin(), raw.end());
    }
    void save(const fs::path& path) const {
        auto f = header(path, h.dump());
        if (!body.empty()) f.write(reinterpret_cast<const char*>(body.data()), body.size());
    }
};
void text(const fs::path& p, const Config& j) { std::ofstream(p) << j.dump(); }
constexpr char embed[] = "model.embed_tokens.weight";
constexpr char weight[] = "model.layers.63.mlp.experts.255.down_proj.weight";

void numeric_and_cache() {
    Temp t;
    Shard a, b;
    a.add(embed, "BF16", {3,2}, words({0x3f80,0xc000,0x4000,0x4080,0x3f00,0xbf00}));
    a.add("half", "F16", {8}, words({0x8000,1,0x3ff,0x400,0x7bff,0xbc00,0xfc00,0x7e00}));
    a.add("lm_head.weight", "F32", {2,3}, floats({1,2,3,-1,0,-2}));
    a.add("conv", "F16", {2,1,2}, words({0x4000,0x3c00,0xc000,0xbe00}));
    constexpr size_t R = 129, C = 130;
    Bytes raw(R*C, 0x38); // 1 everywhere, except the right edge
    for (size_t r = 0; r < R; ++r) raw[r*C + 129] = 0xc0; // -2
    a.add(weight, "F8_E4M3", {R,C}, raw);
    b.add(std::string(weight) + "_scale_inv", "F32", {2,2}, floats({2,3,5,7}));
    b.add("edges.weight", "F8_E4M3FN", {1,8}, {0,0x80,1,0x87,8,0x7e,0xfe,0xff});
    b.add("edges.weight_scale_inv", "F16", {1,1}, words({0x3c00}));
    const char *af = "a.safetensors", *bf = "b.safetensors";
    a.save(t.dir / af); b.save(t.dir / bf);
    Config idx;
    for (const auto& it : a.h.items()) idx["weight_map"][it.key()] = af;
    for (const auto& it : b.h.items()) idx["weight_map"][it.key()] = bf;
    idx["metadata"]["total_size"] = a.body.size() + b.body.size();
    text(t.dir / "model.safetensors.index.json", idx);
    FileWeights w(t.dir, 8);
    check(w.stats().read_ops == 0 && w.stats().shards == 2 && w.stats().tensors == 8, "open read payload");
    check(w.row(embed,3,2,0) == Vec({1,-2}), "BF16 row");
    w.row(embed,3,2,1); w.row(embed,3,2,0); w.row(embed,3,2,2); w.row(embed,3,2,0);
    check(w.stats().cache_hits == 2 && w.stats().evictions == 1 && w.stats().bytes_read == 12, "LRU recency");
    w.row(embed,3,2,1);
    check(w.stats().cache_hits == 2 && w.stats().evictions == 2 && w.stats().cache_bytes == 8, "LRU victim");
    auto h = w.read("half", {8});
    check(std::signbit(h[0]) && h[1] == std::ldexp(1.f,-24) && h[2] == std::ldexp(1023.f,-24) &&
          h[3] == std::ldexp(1.f,-14) && h[4] == 65504 && h[5] == -1 &&
          h[6] == -INFINITY && std::isnan(h[7]), "IEEE binary16 values");
    check(w.read("conv",{2,1,2}) == Vec({2,1,-2,-1.5f}), "3D dense read");
    check(w.linear("lm_head.weight",{2,-1,4},2) == Vec({12,-10}), "F32 matvec");
    check(w.linear(embed,{2,1},3) == Vec({0,8,.5f}), "BF16 matvec");
    auto edge = w.read("edges.weight",{1,8});
    check(!std::signbit(edge[0]) && std::signbit(edge[1]) && edge[2] == 1.f/512 &&
          edge[3] == -7.f/512 && edge[4] == 1.f/64 && edge[5] == 448 &&
          edge[6] == -448 && std::isnan(edge[7]), "E4M3FN values");
    Vec expected(R, 128*2-3);
    expected.back() = 128*5-7;
    check(w.linear(weight, Vec(C,1), R) == expected, "ragged scale grid matvec");
    auto tail = w.row(weight,R,C,128), prev = w.row(weight,R,C,127);
    check(tail[0] == 5 && tail[128] == 7 && tail[129] == -14 &&
          prev[127] == 2 && prev[128] == 3, "FP8 row crossed scale boundary");
    auto all = w.read(weight,{R,C});
    check(all[R*C-1] == -14 && all[C-1] == -6, "FP8 read");
    check(w.stats().peak_cache_bytes <= 8, "cache budget");
    auto reads = w.stats().read_ops;
    rejects([&] { w.row(embed,3,2,3); }, "row index");
    rejects([&] { w.read("conv",{2,2}); }, "same-size shape mismatch");
    rejects([&] { w.linear(weight,Vec(C-1),R); }, "unexpected input width");
    rejects([&] { w.read("absent",{2}); }, "unknown name");
    check(w.stats().read_ops == reads, "invalid request read weights");
    FileWeights nocache(t.dir);
    nocache.row(embed,3,2,0); nocache.row(embed,3,2,0);
    check(nocache.stats().read_ops == 2 && nocache.stats().cache_bytes == 0, "disabled cache");
}

struct Backend : MatvecBackend {
    size_t next = 0, calls = 0, hits = 0;
    std::vector<std::string> keys;
    // All other columns of this fixture are zero.
    std::map<std::string, std::pair<Vec,Vec>> resident;
    bool short_result = false;
    bool multiply_cached(const std::string& key, const std::string& dtype, size_t rows, size_t cols,
                         const Vec& x, Vec& result) override {
        const auto tile = resident.find(key);
        if (tile == resident.end()) return false;
        check(dtype == "F8_E4M3" && rows == tile->second.first.size() && cols == x.size(), "cached descriptor");
        ++hits;
        result.resize(short_result ? 0 : rows);
        for (size_t r = 0; r < result.size(); ++r)
            result[r] = x.front()*tile->second.first[r] + x.back()*tile->second.second[r];
        return true;
    }
    Vec multiply(const std::string& key, const unsigned char* raw, size_t nbytes, const std::string& dtype,
                 const Vec& s, size_t rows, size_t cols, const Vec& x) override {
        if (dtype == "F16") {
            check(s.empty() && nbytes == 4 && rows == 1 && cols == 2 && raw[1] == 0x3c, "dense backend tile");
            return short_result ? Vec{} : Vec{x[0] - x[1]};
        }
        check(key.find(std::string(weight) + ":rows=" + std::to_string(next) + ":" +
                       std::to_string(next + rows)) != std::string::npos, "backend key");
        check(dtype == "F8_E4M3" && next % 128 == 0 && nbytes == rows*cols && nbytes <= (64u<<20), "raw tile");
        size_t stride = (cols+127)/128;
        check(s.size() == (rows+127)/128 * stride, "tile-local scale shape");
        Vec result(rows), left(rows), right(rows);
        for (size_t r = 0; r < rows; ++r) {
            check(raw[r*cols] == 0x38 && raw[(r+1)*cols-1] == 0x40, "FP8 source bytes");
            left[r] = s[r/128*stride];
            right[r] = 2*s[(r/128+1)*stride-1];
            result[r] = x.front()*left[r] + x.back()*right[r];
        }
        resident.emplace(key, std::make_pair(std::move(left),std::move(right)));
        keys.push_back(key); next += rows; ++calls;
        return result;
    }
};

void backend_tiles() {
    Temp t;
    constexpr size_t R=513, C=32769, sr=(R+127)/128, sc=(C+127)/128;
    {
        Shard sh;
        Bytes raw(R*C);
        for (size_t r=0; r<R; ++r) { raw[r*C] = 0x38; raw[(r+1)*C-1] = 0x40; }
        sh.add(weight, "F8_E4M3", {R,C}, raw);
        Vec s(sr*sc);
        for (size_t r=0; r<sr; ++r) for (size_t c=0; c<sc; ++c) s[r*sc+c] = float(r*10+c);
        sh.add(std::string(weight)+"_scale_inv", "F32", {sr,sc}, floats(s));
        sh.add("dense", "F16", {1,2}, words({0x3c00,0xbc00}));
        sh.save(t.dir/"model.safetensors");
    }
    Vec x(C);
    x.front()=2; x.back()=1;
    Backend backend;
    FileWeights w(t.dir, 0, &backend);
    auto y = w.linear(weight,x,R);
    check(backend.calls >= 2 && backend.next == R, "matrix not tiled");
    for (size_t i=0; i<R; ++i) check(y[i] == float((i/128)*40+2*(sc-1)), "wrong tile scale origin");
    check(w.stats().max_read_bytes < R*C, "staged full matrix");
    const auto cold = w.stats();
    check(cold.payload_read_seconds > 0, "payload read timer");
    x.front() = 3; x.back() = -1;
    y = w.linear(weight,x,R);
    for (size_t i=0; i<R; ++i) check(y[i] == float(i/128)*10 - 2*float(sc-1), "cached tile kept stale activation");
    check(backend.hits == backend.calls && cold.read_ops == w.stats().read_ops &&
          cold.bytes_read == w.stats().bytes_read && cold.payload_read_seconds == w.stats().payload_read_seconds &&
          cold.cache_misses == w.stats().cache_misses && w.stats().cache_bytes == 0,
          "resident tile touched host cache or reread scales/weights");
    FileWeights cpu(t.dir);
    check(cpu.linear(weight,x,R) == y && cpu.linear("dense",{4,2},1) == Vec{2}, "CPU tile or binary16 matvec");
    check(w.linear("dense",{4,2},1) == Vec{2}, "dense backend");
    backend.short_result = true;
    rejects([&] { w.linear("dense",{4,2},1); }, "backend output length");
    rejects([&] { w.linear(weight,x,R); }, "cached backend output length");
    backend.short_result = false;
    backend.next=0;
    const size_t previous_calls = backend.calls, previous_hits = backend.hits;
    FileWeights other_reader(t.dir, 0, &backend);
    check(other_reader.linear(weight,x,R) == y && backend.hits == previous_hits &&
          backend.calls == previous_calls*2, "device reused a different reader's key");
}

void malformed() {
    Temp t;
    const auto file = t.dir/"model.safetensors";
    Shard sh;
    sh.add("test", "BF16", {2}, words({0x3f80,0x4000}));
    auto bad = [&](Config h) {
        Shard copy{h, sh.body};
        copy.save(file);
        rejects([&] { FileWeights w(t.dir); }, "bad header");
    };
    auto h = sh.h;
    h["test"]["shape"] = {-1}; bad(h);
    h["test"]["shape"] = {UINT64_MAX,2}; bad(h);
    h["test"]["shape"] = {2.0}; bad(h);
    h["test"]["shape"] = {0}; bad(h);
    h = sh.h; h["test"]["dtype"]="F8_E4M3FNUZ"; bad(h);
    h = sh.h; h["test"]["data_offsets"] = {2,6}; bad(h);
    h = sh.h; h["test"]["data_offsets"] = {2,1}; bad(h);
    h = sh.h; h["test"]["data_offsets"] = {0,2}; bad(h);
    h = sh.h; h["other"] = h.at("test"); bad(h);
    h = sh.h; h["test"]["shape"]={1}; h["test"]["data_offsets"]={2,4}; bad(h);
    h["test"]["data_offsets"]={0,2}; bad(h);
    h = sh.h; h["test"]["dtype"]="F8_E4M3"; h["test"]["shape"]={2,2}; bad(h);
    h = sh.h; h["test_scale_inv"]=h.at("test"); h.erase("test"); bad(h);
    { auto f = header(file, R"({"t":{},"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})"); f.write("1234",4); }
    rejects([&] { FileWeights w(t.dir); }, "duplicate tensor name");
    { std::ofstream f(file,std::ios::binary); f << "12345678"; }
    rejects([&] { FileWeights w(t.dir); }, "unbounded header length");
    sh.save(file);
    fs::resize_file(file,fs::file_size(file)-1);
    rejects([&] { FileWeights w(t.dir); }, "truncated payload");
    sh.save(file);
#ifndef _WIN32
    { // Also detect a short slice read after successful header validation.
        FileWeights w(t.dir);
        fs::resize_file(file, fs::file_size(file)-1);
        rejects([&] { w.read("test",{2}); }, "short read after open");
    }
    sh.save(file);
#endif
    const auto idx=t.dir/"model.safetensors.index.json";
    text(idx, {{"weight_map",{{"test","../x.safetensors"}}}});
    rejects([&] { FileWeights w(t.dir); }, "shard traversal");
    text(idx, {{"weight_map",{{"wrong","model.safetensors"}}}});
    rejects([&] { FileWeights w(t.dir); }, "index disagrees with header");
    text(idx, {{"weight_map",{{"test","model.safetensors"},{"missing","model.safetensors"}}}});
    rejects([&] { FileWeights w(t.dir); }, "missing tensor");
    text(idx, {{"weight_map",{{"test","model.safetensors"}}},{"metadata",{{"total_size",5}}}});
    rejects([&] { FileWeights w(t.dir); }, "wrong index total_size");
    fs::remove(idx);
    Shard grid;
    grid.add("q.weight","F8_E4M3",{1,129},Bytes(129));
    grid.add("q.weight_scale_inv","F32",{2,1},floats({1,1}));
    grid.save(file);
    rejects([&] { FileWeights w(t.dir); }, "transposed scale grid");
    grid.h["q.weight_scale_inv"]["shape"]={1,2};
    grid.save(file);
    {
        Shard nan_scale;
        nan_scale.add("q.weight","F8_E4M3",{1,1},{0x38});
        nan_scale.add("q.weight_scale_inv","F32",{1,1},floats({NAN}));
        nan_scale.save(file);
        FileWeights w(t.dir);
        rejects([&] { w.linear("q.weight",{1},1); }, "non-finite block scale");
    }
    grid.save(file);
    Config cfg = {{"quantization_config",{{"quant_method","fp8"},{"fmt","e4m3"},{"weight_block_size",{128,128}}}}};
    text(t.dir/"config.json",cfg);
    { FileWeights w(t.dir); }
    cfg["quantization_config"]["weight_block_size"]={64,128};
    text(t.dir/"config.json",cfg);
    rejects([&] { FileWeights w(t.dir); }, "unsupported quantization config");
}

void expert_prefetch() {
    Temp t;
    std::string p = "model.layers.3.mlp.experts.55.";
    Shard a,b;
    Config idx;
    for (const char* leaf : {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}) {
        a.add(p+leaf, "F8_E4M3", {2,129}, Bytes(258,0x38));
        b.add(p+leaf+"_scale_inv","F32",{1,2},floats({1,2}));
        idx["weight_map"][p+leaf]="a.safetensors";
        idx["weight_map"][p+leaf+"_scale_inv"]="b.safetensors";
    }
    a.save(t.dir/"a.safetensors"); b.save(t.dir/"b.safetensors");
    text(t.dir/"model.safetensors.index.json",idx);
    FileWeights w(t.dir,32);
    check(w.stats().prefetch_requests == 0, "implicit prefetch during open");
    w.prefetch_expert(p);
    auto s = w.stats();
    check(s.read_ops == 0 && s.bytes_read == 0 && s.payload_read_seconds == 0 &&
          s.cache_bytes == 0 && s.cache_misses == 0, "advice synchronously loaded payload");
    if (s.prefetch_supported)
        check(s.prefetch_requests == 6 && s.prefetch_bytes == 3*(258+8), "incomplete expert read-ahead");
    else
        check(!s.prefetch_requests && !s.prefetch_bytes, "unsupported read-ahead");
    check(w.row(p+"gate_proj.weight",2,129,0).back() == 2 &&
          w.stats().read_ops == 2 && w.stats().cache_hits == 0, "advice altered normal miss path");
    check(w.stats().prefetch_requests == s.prefetch_requests, "implicit prefetch during row");
}

#ifndef _WIN32
void sparse_big_offset() {
    // Last row lives past 4 GiB; Linux/macOS sparse file, only a few disk pages.
    Temp t;
    size_t rows = 300000, cols = 4096;
    uint64_t size = uint64_t(rows) * cols * 4;
    Config h = {{embed,{{"dtype","F32"},{"shape",{rows,cols}},{"data_offsets",{0,size}}}}};
    const std::string p = "model.layers.3.mlp.experts.55.";
    constexpr uint64_t expert_bytes = 80ull << 20;
    h[p+"gate_proj.weight"] = {{"dtype","F32"},{"shape",{5120,4096}},{"data_offsets",{size,size+expert_bytes}}};
    {
        auto f = header(t.dir/"model.safetensors", h.dump());
        f.seekp(static_cast<std::streamoff>(size-4), std::ios::cur);
        auto last = floats({-3.25f});
        f.write(reinterpret_cast<const char*>(last.data()), 4);
        f.seekp(static_cast<std::streamoff>(expert_bytes - 1), std::ios::cur);
        f.put('\0');
    }
    FileWeights w(t.dir);
    rejects([&] { w.read(embed,{rows,cols}); }, "unbounded read");
    auto r = w.row(embed,rows,cols,rows-1);
    check(r.front() == 0 && r.back() == -3.25f && w.stats().bytes_read == cols*4,
          "64-bit row offset and bounded I/O");
    w.prefetch_expert(p);
    if (w.stats().prefetch_supported)
        check(w.stats().prefetch_bytes == (64u<<20) && w.stats().bytes_read == cols*4, "advice byte cap");
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        numeric_and_cache();
        backend_tiles();
        malformed();
        expert_prefetch();
#ifndef _WIN32
        sparse_big_offset();
#endif
        // Optional: validate a local index and its exact shard headers, no payload I/O.
        if (argc == 2) {
            FileWeights w(argv[1]);
            auto s = w.stats();
            std::cout << "checkpoint: " << s.tensors << " tensors, " << s.shards
                      << " shards, " << s.bytes_read << " payload bytes read\n";
        }
        std::cout << "weights_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "weights_test: " << e.what() << '\n';
        return 1;
    }
}
