// High-throughput C example: GSplat encode -> decode using the public C ABI.

#include "../../bindings/c/spatial_codecs_interops.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void assert_f32_close(const float a, const float b, const float eps) {
    const float d = fabsf(a - b);
    if (!(d <= eps)) {
        fprintf(stderr, "float mismatch: %f vs %f (|d|=%f, eps=%f)\n", a, b, d, eps);
        assert(0);
    }
}

static pcencodingparamsopaque* parse_gsplat_params_or_abort(void) {
    // Custom params example (mirrors patterns in bench.toml).
    const char* toml =
        "Zstd = { level = 3, inner = { Ply = { encoding = 'BinaryLittleEndian' } } }";

    pcencodingparamsopaque* prms = NULL;
    const bool ok = pc_encoding_params_from_toml((const uint8_t*)toml, (uint64_t)strlen(toml), &prms);
    if (!ok || prms == NULL) {
        const int8_t* msg = pc_last_error_message();
        fprintf(stderr, "pc_encoding_params_from_toml failed: %s\n", msg ? (const char*)msg : "<no message>");
        assert(0);
    }
    return prms;
}

int main(void) {
    // ----- Input (2 splats) -----
    const float pos[2 * 3] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 3.0f,
    };
    const float col[2 * 4] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 0.8f,
    };
    const float scale[2 * 3] = {
        1.0f, 1.0f, 1.0f,
        0.5f, 0.75f, 1.25f,
    };
    const float rot[2 * 4] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.70710678f, 0.0f, 0.70710678f, 0.0f,
    };

    // ----- Encoder setup -----
    void *enc = pc_gsplat_encoder_create();
    assert(enc != NULL);

    pcencodingparamsopaque* params = parse_gsplat_params_or_abort();
    assert(params != NULL);

    // Note: `pcgsplatoutbuffers` is used for both input (slice*) and output (mutslice*).
    const pcgsplatoutbuffers in = (pcgsplatoutbuffers){
        .positions = (pcmutslicevec3){.data = pos, .len = 2},
        .colors = (pcmutslicevec4){.data = col, .len = 2},
        .scales = (pcmutslicevec3){.data = scale, .len = 2},
        .rotations = (pcmutslicevec4){.data = rot, .len = 2},
    };

    pcbyteslice bytes = (pcbyteslice){0};
    const bool ok = pc_gsplat_encode_with_params(enc, in, params, &bytes);
    assert(ok);
    assert(bytes.data != NULL);
    assert(bytes.len > 0);

    // ----- Decoder setup -----
    void *dec = NULL;
    pcstatus st = pc_gsplat_decoder_create(&dec);
    assert(dec != NULL);

    pcgsplatmeta meta = (pcgsplatmeta){0};
    st = pc_gsplat_decoder_decode_from_bytes(dec, (sliceu8){.data = bytes.data, .len = bytes.len}, &meta);
    assert(meta.splat_count == 2);

    const size_t n = (size_t)meta.splat_count * 3;

    float pos_out[n * 3];
    float col_out[n * 4];
    float scale_out[n * 3];
    float rot_out[n * 4];
    
    memset(pos_out, 0, sizeof(pos_out));
    memset(col_out, 0, sizeof(col_out));
    memset(scale_out, 0, sizeof(scale_out));
    memset(rot_out, 0, sizeof(rot_out));

    pcgsplatoutbuffers out = (pcgsplatoutbuffers){
        .positions = (pcmutslicevec3){.data = pos_out, .len = n},
        .colors = (pcmutslicevec4){.data = col_out, .len = n},
        .scales = (pcmutslicevec3){.data = scale_out, .len = n},
        .rotations = (pcmutslicevec4){.data = rot_out, .len = n},
    };

    st = pc_gsplat_decoder_write_to_buffers(dec, out, &meta);
    assert(meta.splat_count == n);

    // ----- Verify (EPS compare) -----
    const float eps = 1e-6f;
    for (size_t i = 0; i < 2 * 3; i++) {
        assert_f32_close(pos[i], pos_out[i], eps);
        assert_f32_close(scale[i], scale_out[i], eps);
    }
    for (size_t i = 0; i < 2 * 4; i++) {
        assert_f32_close(col[i], col_out[i], eps);
        assert_f32_close(rot[i], rot_out[i], eps);
    }

    printf("OK: gsplat roundtrip (%zu bytes)\n", (size_t)bytes.len);

    pc_gsplat_decoder_free(dec);
    pc_gsplat_encoder_free(enc);
    pc_encoding_params_free(params);
    return 0;
}
