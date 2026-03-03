// High-throughput C++ example: GSplat encode -> decode using the public C ABI.
//
// Showcases:
// - Encoding params via TOML (see bench.toml for patterns)
// - AoS buffers (pcvec3 / pcvec4 arrays) passed via slice structs
// - Two-step decoding:
//     1) pc_gsplat_decoder_decode_from_bytes -> pcgsplatmeta.splat_count
//     2) pc_gsplat_decoder_write_to_buffers -> copy into caller-provided buffers

#include "../../bindings/c/spatial_codecs_interops.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static void assert_close_f32(float a, float b, float eps) {
    const float d = std::fabs(a - b);
    if (!(d <= eps)) {
        std::cerr << "float mismatch: " << a << " vs " << b
                  << " (|d|=" << d << ", eps=" << eps << ")\n";
        std::abort();
    }
}

static void die_status(const char* what, pcstatus st) {
    const auto* msg = pc_last_error_message();
    std::cerr << what << " failed (status=" << static_cast<int>(st) << "): "
              << (msg ? reinterpret_cast<const char*>(msg) : "<no message>") << "\n";
    std::abort();
}

static pcencodingparamsopaque* parse_gsplat_params_or_die() {
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

int main() {
    // ----- Input (AoS; 2 splats) -----
    std::vector<pcvec3> pos = {
        pcvec3{0.0f, 0.0f, 0.0f},
        pcvec3{1.0f, 2.0f, 3.0f},
    };
    std::vector<pcvec4> col = {
        pcvec4{1.0f, 0.0f, 0.0f, 1.0f},
        pcvec4{0.0f, 1.0f, 0.0f, 0.8f},
    };
    std::vector<pcvec3> scale = {
        pcvec3{1.0f, 1.0f, 1.0f},
        pcvec3{0.5f, 0.75f, 1.25f},
    };
    std::vector<pcvec4> rot = {
        pcvec4{1.0f, 0.0f, 0.0f, 0.0f},
        pcvec4{0.70710678f, 0.0f, 0.70710678f, 0.0f},
    };

    const std::size_t splat_count_in = pos.size();
    assert(col.size() == splat_count_in);
    assert(scale.size() == splat_count_in);
    assert(rot.size() == splat_count_in);

    // ----- Encoder setup -----
    void* enc = pc_gsplat_encoder_create();
    assert(enc != nullptr);

    pcencodingparamsopaque* params = parse_gsplat_params_or_die();

    // Note: `pcgsplatoutbuffers` uses mutable slices for historical reasons;
    // the encoder treats these as read-only.
    const pcgsplatoutbuffers in = pcgsplatoutbuffers{
        /* positions  */ pcmutslicevec3{pos.data(), static_cast<std::uint64_t>(pos.size())},
        /* colors     */ pcmutslicevec4{col.data(), static_cast<std::uint64_t>(col.size())},
        /* scales     */ pcmutslicevec3{scale.data(), static_cast<std::uint64_t>(scale.size())},
        /* rotations  */ pcmutslicevec4{rot.data(), static_cast<std::uint64_t>(rot.size())},
    };

    pcbyteslice encoded{};
    const bool ok = pc_gsplat_encode_with_params(enc, in, params, &encoded);
    assert(ok);
    assert(encoded.data != nullptr);
    assert(encoded.len > 0);

    // ----- Decoder setup -----
    void* dec = nullptr;
    pcstatus st = pc_gsplat_decoder_create(&dec);
    if (st != PCSTATUS_OK || dec == nullptr) {
        die_status("pc_gsplat_decoder_create", st);
    }

    // ----- Two-step decode (dynamic sizing) -----
    pcgsplatmeta meta{};
    st = pc_gsplat_decoder_decode_from_bytes(
        dec,
        sliceu8{reinterpret_cast<const std::uint8_t*>(encoded.data), encoded.len},
        &meta);
    if (st != PCSTATUS_OK) {
        die_status("pc_gsplat_decoder_decode_from_bytes", st);
    }

    const std::size_t splat_count = static_cast<std::size_t>(meta.splat_count);
    assert(splat_count == splat_count_in);

    std::vector<pcvec3> pos_out(splat_count);
    std::vector<pcvec4> col_out(splat_count);
    std::vector<pcvec3> scale_out(splat_count);
    std::vector<pcvec4> rot_out(splat_count);

    const pcgsplatoutbuffers out = pcgsplatoutbuffers{
        /* positions */ pcmutslicevec3{pos_out.data(), static_cast<std::uint64_t>(pos_out.size())},
        /* colors    */ pcmutslicevec4{col_out.data(), static_cast<std::uint64_t>(col_out.size())},
        /* scales    */ pcmutslicevec3{scale_out.data(), static_cast<std::uint64_t>(scale_out.size())},
        /* rotations */ pcmutslicevec4{rot_out.data(), static_cast<std::uint64_t>(rot_out.size())},
    };

    st = pc_gsplat_decoder_write_to_buffers(dec, out, &meta);
    if (st != PCSTATUS_OK) {
        die_status("pc_gsplat_decoder_write_to_buffers", st);
    }
    assert(static_cast<std::size_t>(meta.splat_count) == splat_count);

    // ----- Verify -----
    constexpr float eps = 1e-6f;
    for (std::size_t i = 0; i < splat_count; i++) {
        assert_close_f32(pos[i].x, pos_out[i].x, eps);
        assert_close_f32(pos[i].y, pos_out[i].y, eps);
        assert_close_f32(pos[i].z, pos_out[i].z, eps);

        assert_close_f32(scale[i].x, scale_out[i].x, eps);
        assert_close_f32(scale[i].y, scale_out[i].y, eps);
        assert_close_f32(scale[i].z, scale_out[i].z, eps);

        assert_close_f32(col[i].x, col_out[i].x, eps);
        assert_close_f32(col[i].y, col_out[i].y, eps);
        assert_close_f32(col[i].z, col_out[i].z, eps);
        assert_close_f32(col[i].w, col_out[i].w, eps);

        assert_close_f32(rot[i].x, rot_out[i].x, eps);
        assert_close_f32(rot[i].y, rot_out[i].y, eps);
        assert_close_f32(rot[i].z, rot_out[i].z, eps);
        assert_close_f32(rot[i].w, rot_out[i].w, eps);
    }

    std::cout << "OK: gsplat roundtrip (" << encoded.len << " bytes)\n";

    pc_gsplat_decoder_free(dec);
    pc_gsplat_encoder_free(enc);
    pc_encoding_params_free(params);
    return 0;
}
