// High-throughput C# example: Points + GSplat roundtrip using the public C# bindings.
//
// Showcases:
// - Custom encoding params via TOML (see bench.toml for patterns)
// - Allocation-light usage: arrays allocated once; params parsed once
// - Encoder returns an unmanaged view (no copy)
// - Points decode uses the two-step point decoder API (meta -> sized buffers)
//
// Notes:
// - Some codecs may reorder points; this sample verifies points order-independently.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Be.Ugent;

internal static class Program
{
    private static uint ColorKey(byte r, byte g, byte b) => ((uint)r << 16) | ((uint)g << 8) | b;

    private static (uint, uint, uint) CoordKey(float x, float y, float z)
        => (unchecked((uint)BitConverter.SingleToInt32Bits(x)), unchecked((uint)BitConverter.SingleToInt32Bits(y)), unchecked((uint)BitConverter.SingleToInt32Bits(z)));

    private static void SortPoints(List<((float x, float y, float z) pos, (byte r, byte g, byte b) col)> pts)
    {
        pts.Sort((a, b) =>
        {
            var ca = ColorKey(a.col.r, a.col.g, a.col.b);
            var cb = ColorKey(b.col.r, b.col.g, b.col.b);
            var c = ca.CompareTo(cb);
            if (c != 0) return c;
            return CoordKey(a.pos.x, a.pos.y, a.pos.z).CompareTo(CoordKey(b.pos.x, b.pos.y, b.pos.z));
        });
    }

    private static void AssertClose(float a, float b, float eps, string msg)
    { 
        var d = MathF.Abs(a - b);
        if (d > eps) throw new Exception($"{msg}: {a} vs {b} (|d|={d})");
    }

    private static string LastErrorMessage()
    {
        var ptr = SpatialCodecsInterop.pc_last_error_message();
        return ptr == IntPtr.Zero ? "<no message>" : (Marshal.PtrToStringAnsi(ptr) ?? "<unreadable>");
    }

    private static unsafe IntPtr ParseParamsOrThrow(string toml)
    {
        if (toml is null) throw new ArgumentNullException(nameof(toml));

        var bytes = Encoding.UTF8.GetBytes(toml);
        if (bytes.Length == 0) throw new ArgumentException("TOML must not be empty", nameof(toml));

        IntPtr paramsPtr = IntPtr.Zero;
        fixed (byte* p = bytes)
        {
            var ok = SpatialCodecsInterop.pc_encoding_params_from_toml(ref p[0], (ulong)bytes.Length, ref paramsPtr);
            if (!ok || paramsPtr == IntPtr.Zero)
                throw new Exception($"pc_encoding_params_from_toml failed: {LastErrorMessage()}");
        }

        return paramsPtr;
    }

    private static unsafe void RoundtripPoints()
    {
        const int nIn = 4;

        var pos = new float[nIn * 3]
        {
            0f, 0f, 0f,
            1f, 2f, 3f,
            -4f, 5f, -6f,
            10f, -2.5f, 0.125f,
        };
        var col = new byte[nIn * 3]
        {
            255, 0, 0,
            0, 255, 0,
            0, 0, 255,
            123, 45, 67,
        };

        var enc = SpatialCodecsInterop.pc_point_encoder_create();
        if (enc == IntPtr.Zero) throw new Exception("pc_point_encoder_create failed");

        // Custom params example.
        // Equivalent to: Ply wrapped in Zstd(level=3).
        var paramsPtr = ParseParamsOrThrow(
            "Zstd = { level = 3, inner = { Ply = { encoding = 'BinaryLittleEndian' } } }"
        );

        PcByteSlice encoded;
        fixed (float* posPtr = pos)
        fixed (byte* colPtr = col)
        {
            var input = new PcPointInBuffers
            {
                positions = new PcSliceF32 { data = (IntPtr)posPtr, len = (ulong)pos.Length },
                colors = new PcSliceU8 { data = (IntPtr)colPtr, len = (ulong)col.Length },
            };

            var ok = SpatialCodecsInterop.pc_point_encode_with_params(enc, input, paramsPtr, out encoded);
            if (!ok) throw new Exception($"pc_point_encode_with_params failed: {LastErrorMessage()}");
        }

        if (encoded.data == IntPtr.Zero || encoded.len == 0)
            throw new Exception("encode returned empty bytes");

        var dec = SpatialCodecsInterop.pc_point_decoder_create();
        if (dec == IntPtr.Zero) throw new Exception("pc_point_decoder_create failed");

        // ----- Two-step decode (dynamic sizing) -----
        var st = SpatialCodecsInterop.pc_point_decoder_decode_from_bytes(dec, new Sliceu8(encoded.data, encoded.len), out var meta);
        if (st != PcStatus.Ok) throw new Exception($"pc_point_decoder_decode_from_bytes failed: {st} ({LastErrorMessage()})");

        var n = checked((int)meta.point_count);
        if (n != nIn) throw new Exception($"unexpected point count: {n} (expected {nIn})");

        // Output buffers are flattened (x,y,z per point).
        var posOut = new float[checked(n * 3)];
        var colOut = new byte[checked(n * 3)];

        fixed (float* posOutPtr = posOut)
        fixed (byte* colOutPtr = colOut)
        {
            var outBuf = new PcPointOutBuffers
            {
                positions = new PcMutSliceF32 { data = (IntPtr)posOutPtr, len = (ulong)posOut.Length },
                colors = new PcMutSliceU8 { data = (IntPtr)colOutPtr, len = (ulong)colOut.Length },
            };

            st = SpatialCodecsInterop.pc_point_decoder_write_to_buffers(dec, outBuf, out meta);
            if (st != PcStatus.Ok) throw new Exception($"pc_point_decoder_write_to_buffers failed: {st} ({LastErrorMessage()})");
        }

        if (checked((int)meta.point_count) != n) throw new Exception($"meta.point_count changed unexpectedly: {meta.point_count}");

        // ----- Verify (order-independent) -----
        var original = new List<((float x, float y, float z) pos, (byte r, byte g, byte b) col)>(n);
        var decoded = new List<((float x, float y, float z) pos, (byte r, byte g, byte b) col)>(n);

        for (var i = 0; i < n; i++)
        {
            original.Add(((pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]), (col[i * 3 + 0], col[i * 3 + 1], col[i * 3 + 2])));
            decoded.Add(((posOut[i * 3 + 0], posOut[i * 3 + 1], posOut[i * 3 + 2]), (colOut[i * 3 + 0], colOut[i * 3 + 1], colOut[i * 3 + 2])));
        }

        SortPoints(original);
        SortPoints(decoded);

        for (var i = 0; i < n; i++)
        {
            if (!original[i].col.Equals(decoded[i].col))
                throw new Exception($"color mismatch at {i}: {original[i].col} vs {decoded[i].col}");
        }

        const float eps = 1e-4f;
        for (var i = 0; i < n; i++)
        {
            AssertClose(original[i].pos.x, decoded[i].pos.x, eps, $"pos[{i}].x");
            AssertClose(original[i].pos.y, decoded[i].pos.y, eps, $"pos[{i}].y");
            AssertClose(original[i].pos.z, decoded[i].pos.z, eps, $"pos[{i}].z");
        }

        SpatialCodecsInterop.pc_point_decoder_free(dec);
        SpatialCodecsInterop.pc_point_encoder_free(enc);
        SpatialCodecsInterop.pc_encoding_params_free(paramsPtr);

        Console.WriteLine($"OK: points roundtrip ({encoded.len} bytes)");
    }

    private static unsafe void RoundtripGsplat()
    {
        const int nIn = 2;

        // AoS arrays matching the ABI layout.
        var pos = new PcVec3[nIn]
        {
            new PcVec3 { x = 0f, y = 0f, z = 0f },
            new PcVec3 { x = 1f, y = 2f, z = 3f },
        };
        var col = new PcVec4[nIn]
        {
            new PcVec4 { x = 1f, y = 0f, z = 0f, w = 1f },
            new PcVec4 { x = 0f, y = 1f, z = 0f, w = 0.8f },
        };
        var scale = new PcVec3[nIn]
        {
            new PcVec3 { x = 1f, y = 1f, z = 1f },
            new PcVec3 { x = 0.5f, y = 0.75f, z = 1.25f },
        };
        var rot = new PcVec4[nIn]
        {
            new PcVec4 { x = 1f, y = 0f, z = 0f, w = 0f },
            new PcVec4 { x = 0.70710678f, y = 0f, z = 0.70710678f, w = 0f },
        };

        var enc = SpatialCodecsInterop.pc_gsplat_encoder_create();
        if (enc == IntPtr.Zero) throw new Exception("pc_gsplat_encoder_create failed");

        // Custom params example (mirrors patterns in bench.toml).
        var paramsPtr = ParseParamsOrThrow(
            "Zstd = { level = 3, inner = { Ply = { encoding = 'BinaryLittleEndian' } } }"
        );

        PcByteSlice encoded;
        fixed (PcVec3* posPtr = pos)
        fixed (PcVec4* colPtr = col)
        fixed (PcVec3* scalePtr = scale)
        fixed (PcVec4* rotPtr = rot)
        {
            // Note: PcGsplatOutBuffers uses mutable slices; encoder treats them as read-only.
            var input = new PcGsplatOutBuffers
            {
                positions = new PcMutSliceVec3 { data = (IntPtr)posPtr, len = (ulong)nIn },
                colors = new PcMutSliceVec4 { data = (IntPtr)colPtr, len = (ulong)nIn },
                scales = new PcMutSliceVec3 { data = (IntPtr)scalePtr, len = (ulong)nIn },
                rotations = new PcMutSliceVec4 { data = (IntPtr)rotPtr, len = (ulong)nIn },
            };

            var ok = SpatialCodecsInterop.pc_gsplat_encode_with_params(enc, input, paramsPtr, out encoded);
            if (!ok) throw new Exception($"pc_gsplat_encode_with_params failed: {LastErrorMessage()}");
        }

        if (encoded.data == IntPtr.Zero || encoded.len == 0)
            throw new Exception("gsplat encode returned empty bytes");

        // Decoder create returns status + out handle.
        IntPtr dec = IntPtr.Zero;
        var st = SpatialCodecsInterop.pc_gsplat_decoder_create(ref dec);
        if (st != PcStatus.Ok || dec == IntPtr.Zero)
            throw new Exception($"pc_gsplat_decoder_create failed: {st}");

        // Decode to internal vec first.
        st = SpatialCodecsInterop.pc_gsplat_decoder_decode_from_bytes(dec, new Sliceu8(encoded.data, encoded.len), out var meta);
        if (st != PcStatus.Ok) throw new Exception($"pc_gsplat_decoder_decode_from_bytes failed: {st}");
        var n = checked((int)meta.splat_count);
        if (n != nIn) throw new Exception($"unexpected splat count: {n} (expected {nIn})");

        var posOut = new PcVec3[n];
        var colOut = new PcVec4[n];
        var scaleOut = new PcVec3[n];
        var rotOut = new PcVec4[n];

        fixed (PcVec3* posOutPtr = posOut)
        fixed (PcVec4* colOutPtr = colOut)
        fixed (PcVec3* scaleOutPtr = scaleOut)
        fixed (PcVec4* rotOutPtr = rotOut)
        {
            var outBuf = new PcGsplatOutBuffers
            {
                positions = new PcMutSliceVec3 { data = (IntPtr)posOutPtr, len = (ulong)n },
                colors = new PcMutSliceVec4 { data = (IntPtr)colOutPtr, len = (ulong)n },
                scales = new PcMutSliceVec3 { data = (IntPtr)scaleOutPtr, len = (ulong)n },
                rotations = new PcMutSliceVec4 { data = (IntPtr)rotOutPtr, len = (ulong)n },
            };

            st = SpatialCodecsInterop.pc_gsplat_decoder_write_to_buffers(dec, outBuf, out meta);
            if (st != PcStatus.Ok) throw new Exception($"pc_gsplat_decoder_write_to_buffers failed: {st}");
        }

        const float eps = 1e-6f;
        for (var i = 0; i < n; i++)
        {
            AssertClose(pos[i].x, posOut[i].x, eps, $"pos[{i}].x");
            AssertClose(pos[i].y, posOut[i].y, eps, $"pos[{i}].y");
            AssertClose(pos[i].z, posOut[i].z, eps, $"pos[{i}].z");

            AssertClose(scale[i].x, scaleOut[i].x, eps, $"scale[{i}].x");
            AssertClose(scale[i].y, scaleOut[i].y, eps, $"scale[{i}].y");
            AssertClose(scale[i].z, scaleOut[i].z, eps, $"scale[{i}].z");

            AssertClose(col[i].x, colOut[i].x, eps, $"col[{i}].x");
            AssertClose(col[i].y, colOut[i].y, eps, $"col[{i}].y");
            AssertClose(col[i].z, colOut[i].z, eps, $"col[{i}].z");
            AssertClose(col[i].w, colOut[i].w, eps, $"col[{i}].w");

            AssertClose(rot[i].x, rotOut[i].x, eps, $"rot[{i}].x");
            AssertClose(rot[i].y, rotOut[i].y, eps, $"rot[{i}].y");
            AssertClose(rot[i].z, rotOut[i].z, eps, $"rot[{i}].z");
            AssertClose(rot[i].w, rotOut[i].w, eps, $"rot[{i}].w");
        }

        SpatialCodecsInterop.pc_gsplat_decoder_free(dec);
        SpatialCodecsInterop.pc_gsplat_encoder_free(enc);
        SpatialCodecsInterop.pc_encoding_params_free(paramsPtr);

        Console.WriteLine($"OK: gsplat roundtrip ({encoded.len} bytes)");
    }

    public static void Main()
    {
        RoundtripPoints();
        RoundtripGsplat();
    }
}
