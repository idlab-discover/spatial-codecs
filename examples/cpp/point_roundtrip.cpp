// High-throughput C++ example: Point encode -> decode using the C ABI.
//
// Showcases:
// - Encoding params via TOML (see bench.toml for patterns)
// - Two-step decoding:
//     1) pc_point_decoder_decode_from_bytes -> pcpointmeta.point_count
//     2) pc_point_decoder_write_to_buffers -> copy into caller-provided buffers

#include "spatial_codecs_interops.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

static void assert_close(float a, float b, float eps) {
    const float d = std::fabs(a - b);
    if (!(d <= eps)) {
        std::cerr << "float mismatch: " << a << " vs " << b << " (|d|=" << d << ")\n";
        std::abort();
    }
}

static pcencodingparamsopaque* parse_point_params_or_die() {
    // Custom params example (mirrors patterns in bench.toml).
    const char* toml =
        "Zstd = { level = 3, inner = { Ply = { encoding = 'BinaryLittleEndian' } } }";

    pcencodingparamsopaque* prms = nullptr;
    const bool ok = pc_encoding_params_from_toml(
        reinterpret_cast<const std::uint8_t*>(toml),
        static_cast<std::uint64_t>(std::strlen(toml)),
        &prms);

    if (!ok || prms == nullptr) {
        const auto* msg = pc_last_error_message();
        std::cerr << "pc_encoding_params_from_toml failed: "
                  << (msg ? reinterpret_cast<const char*>(msg) : "<no message>")
                  << "\n";
        std::abort();
    }

    return prms;
}

static void die_status(const char* what, pcstatus st) {
    const auto* msg = pc_last_error_message();
    std::cerr << what << " failed (status=" << static_cast<int>(st) << "): "
              << (msg ? reinterpret_cast<const char*>(msg) : "<no message>") << "\n";
    std::abort();
}

int main() {
    const float coords[4 * 3] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 3.0f,
        -4.0f, 5.0f, -6.0f,
        10.0f, -2.5f, 0.125f,
    };
    const std::uint8_t colors[4 * 3] = {
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        123, 45, 67,
    };

    void* enc = pc_point_encoder_create();
    assert(enc != nullptr);

    pcencodingparamsopaque* params = parse_point_params_or_die();

    const pcpointinbuffers in = {
        /* positions */ (pcslicef32){coords, 4 * 3},
        /* colors */ (pcsliceu8){colors, 4 * 3},
    };

    pcbyteslice encoded{};
    const bool ok = pc_point_encode_with_params(enc, in, params, &encoded);
    assert(ok);
    assert(encoded.data != nullptr);
    assert(encoded.len > 0);

    void* dec = pc_point_decoder_create();
    assert(dec != nullptr);

    // ----- Two-step decode (dynamic sizing) -----
    pcpointmeta meta{};
    pcstatus st = pc_point_decoder_decode_from_bytes(
        dec,
        (sliceu8){reinterpret_cast<const std::uint8_t*>(encoded.data), encoded.len},
        &meta);
    if (st != PCSTATUS_OK) {
        die_status("pc_point_decoder_decode_from_bytes", st);
    }

    const std::size_t point_count = static_cast<std::size_t>(meta.point_count);
    const std::size_t n = point_count * 3;

    std::vector<float> coords_out(n, 0.0f);
    std::vector<std::uint8_t> colors_out(n, 0);

    const pcpointoutbuffers out = {
        (pcmutslicef32){coords_out.data(), static_cast<std::uint64_t>(coords_out.size())},
        (pcmutsliceu8){colors_out.data(), static_cast<std::uint64_t>(colors_out.size())},
    };

    st = pc_point_decoder_write_to_buffers(dec, out, &meta);
    if (st != PCSTATUS_OK) {
        die_status("pc_point_decoder_write_to_buffers", st);
    }

    assert(meta.point_count == 4);

    constexpr float eps = 1e-4f;
    for (std::size_t i = 0; i < n; i++) {
        assert_close(coords[i], coords_out[i], eps);
        assert(colors[i] == colors_out[i]);
    }

    std::cout << "OK: point roundtrip (" << encoded.len << " bytes)\n";

    pc_point_decoder_free(dec);
    pc_point_encoder_free(enc);
    pc_encoding_params_free(params);
    return 0;
}
