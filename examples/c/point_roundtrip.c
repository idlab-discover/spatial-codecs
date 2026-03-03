// High-throughput C example: Point encode -> decode using the public C ABI.
//
// This example showcases:
// - Encoding params via TOML (see bench.toml for patterns)
// - Encode using pc_point_encode_with_params
// - Two-step decoding:
//     1) pc_point_decoder_decode_from_bytes -> pcpointmeta.point_count
//     2) pc_point_decoder_write_to_buffers -> copy into caller-provided buffers
//
// Notes:
// - The encoder owns the returned pcbyteslice buffer; it remains valid until the next
//   encode call on the same encoder (or encoder destruction).
// - Some codecs may reorder points. This sample uses an element-wise comparison for
//   simplicity; if you enable reorder-capable codecs, compare as a multiset.

#include "spatial_codecs_interops.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void assert_f32_close(const float a, const float b, const float eps) {
    const float d = fabsf(a - b);
    if (!(d <= eps)) {
        fprintf(stderr, "float mismatch: %f vs %f (|d|=%f, eps=%f)\n", a, b, d, eps);
        assert(0);
    }
}

static pcencodingparamsopaque* parse_point_params_or_abort(void) {
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

static void* checked_malloc(size_t nbytes) {
    if (nbytes == 0) {
        return NULL;
    }
    void* p = malloc(nbytes);
    if (!p) {
        fprintf(stderr, "malloc(%zu) failed\n", nbytes);
        assert(0);
    }
    return p;
}

int main(void) {
    const float coords[4 * 3] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 3.0f,
        -4.0f, 5.0f, -6.0f,
        10.0f, -2.5f, 0.125f,
    };
    const uint8_t colors[4 * 3] = {
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        123, 45, 67,
    };

    void* enc = pc_point_encoder_create();
    assert(enc != NULL);

    pcencodingparamsopaque* params = parse_point_params_or_abort();

    const pcpointinbuffers in = {
        .positions = (pcslicef32){.data = coords, .len = 4 * 3},
        .colors = (pcsliceu8){.data = colors, .len = 4 * 3},
    };

    pcbyteslice encoded = (pcbyteslice){0};
    const bool ok = pc_point_encode_with_params(enc, in, params, &encoded);
    assert(ok);
    assert(encoded.data != NULL);
    assert(encoded.len > 0);

    void* dec = pc_point_decoder_create();
    assert(dec != NULL);

    // ----- Two-step decode (dynamic sizing) -----
    pcpointmeta meta = (pcpointmeta){0};
    pcstatus st = pc_point_decoder_decode_from_bytes(
        dec,
        (sliceu8){.data = encoded.data, .len = encoded.len},
        &meta);

    if (st != PCSTATUS_OK) {
        const int8_t* msg = pc_last_error_message();
        fprintf(stderr, "pc_point_decoder_decode_from_bytes failed (status=%d): %s\n",
                (int)st, msg ? (const char*)msg : "<no message>");
        assert(0);
    }

    assert(meta.point_count == 4);

    const size_t n = (size_t)meta.point_count * 3;

    float* coords_out = (float*)checked_malloc(n * sizeof(float));
    uint8_t* colors_out = (uint8_t*)checked_malloc(n * sizeof(uint8_t));

    const pcpointoutbuffers out = {
        .positions = (pcmutslicef32){.data = coords_out, .len = (uint64_t)n},
        .colors = (pcmutsliceu8){.data = colors_out, .len = (uint64_t)n},
    };

    st = pc_point_decoder_write_to_buffers(dec, out, &meta);
    if (st != PCSTATUS_OK) {
        const int8_t* msg = pc_last_error_message();
        fprintf(stderr, "pc_point_decoder_write_to_buffers failed (status=%d): %s\n",
                (int)st, msg ? (const char*)msg : "<no message>");
        assert(0);
    }

    assert(meta.point_count == 4);

    const float eps = 1e-4f;
    for (size_t i = 0; i < n; i++) {
        assert_f32_close(coords[i], coords_out[i], eps);
        assert(colors[i] == colors_out[i]);
    }

    printf("OK: point roundtrip (%zu bytes)\n", (size_t)encoded.len);

    free(coords_out);
    free(colors_out);

    pc_point_decoder_free(dec);
    pc_point_encoder_free(enc);
    pc_encoding_params_free(params);
    return 0;
}
