// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.Buffers;

internal static class NativeBufferHelpers
{
    public const int AvxVectorSizeFp32 = 8;
    public const int AvxVectorSizeFp64 = 4;
    public const int L1Size = 64 * 1024;
    public const int HalfL1Size = L1Size / 2;
    public const int TileSizeFp32 = L1Size / sizeof(float);
    public const int TileSizeFp64 = L1Size / sizeof(double);
    public const int HalfTileSizeFp32 = TileSizeFp32 / 2;
    public const int HalfTileSizeFp64 = TileSizeFp64 / 2;

    [MethodImpl(ImplementationOptions.HotPath)]
    public static int GetTileSize<TNumber>()
        where TNumber : unmanaged
    {
        return L1Size / Unsafe.SizeOf<TNumber>();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static unsafe void Pack1d<TNumber>(TNumber* src, TNumber* dst, long n)
        where TNumber : unmanaged
    {
        for (int i = 0; i < n; i++)
        {
            dst[i] = src[i];
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static unsafe void Pack1dFp32Avx(float* src, float* dst, long n)
    {
        long i = 0;
        for (; i <= n - AvxVectorSizeFp32; i += AvxVectorSizeFp32)
        {
            var vec = Avx.LoadVector256(src + i);
            Avx.Store(dst + i, vec);
        }

        for (; i < n; i++)
        {
            dst[i] = src[i];
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static unsafe void Pack1dFp64Avx(double* src, double* dst, long n)
    {
        long i = 0;
        for (; i <= n - AvxVectorSizeFp64; i += AvxVectorSizeFp64)
        {
            var vec = Avx.LoadVector256(src + i);
            Avx.Store(dst + i, vec);
        }

        for (; i < n; i++)
        {
            dst[i] = src[i];
        }
    }
}