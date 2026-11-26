#pragma once

#include "weights.hpp"

namespace strata::gigachat4 {

// Uses the current CUDA device. Calls are synchronous; the byte budget is for
// resident raw weights and their scales. Zero selects streaming only. Opt in
// to double-buffered H2D for multi-tile, whole-row streaming with overlap=true.
std::unique_ptr<MatvecBackend> make_cuda_backend(size_t cache_bytes, bool overlap = false);

// Cumulative counters and CUDA-event times (ms); see docs/gigachat4/CUDA.md.
Config cuda_metrics(const MatvecBackend&);

namespace detail {
// E4M3FN (not FNUZ): ±0, subnormals in units of 2^-9, ±448, two NaNs.
// Kept available to the device test: adding a decoded -0 to a dot product's
// +0 accumulator legitimately loses its sign and cannot test that encoding.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr uint32_t e4m3fn_f32_bits(unsigned char byte) {
    const uint32_t sign = uint32_t(byte & 0x80u) << 24;
    const uint32_t exp = (byte >> 3) & 15u, frac = byte & 7u;
    if (exp == 15 && frac == 7) return sign | 0x7fc00000u;
    if (exp != 0) return sign | ((exp + 120u) << 23) | (frac << 20);
    if (frac == 0) return sign;
    const unsigned top = frac >= 4 ? 2 : frac >= 2 ? 1 : 0;
    return sign | ((118u + top) << 23) | ((frac - (1u << top)) << (23 - top));
}
} // namespace detail
} // namespace strata::gigachat4
