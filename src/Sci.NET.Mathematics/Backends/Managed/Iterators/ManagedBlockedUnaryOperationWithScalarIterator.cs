// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Performance;
using Sci.NET.Mathematics.Backends.Managed.Buffers;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedBlockedUnaryOperationWithScalarIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* inputPtr, TNumber* resultPtr, TNumber scalar, long n)
        where TOp : IUnaryOperationWithScalar<TNumber>, IUnaryOperationWithScalarAvx, IUnaryOperationWithScalarAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (TOp.IsAvxFmaSupported())
        {
            switch (TNumber.Zero)
            {
                case float:
                    ForAvxFma<TOp>(
                        (float*)inputPtr,
                        (float*)resultPtr,
                        float.CreateChecked(scalar),
                        n);
                    return;
                case double:
                    ForAvxFma<TOp>(
                        (double*)inputPtr,
                        (double*)resultPtr,
                        double.CreateChecked(scalar),
                        n);
                    return;
            }
        }

        if (TOp.IsAvxSupported())
        {
            switch (TNumber.Zero)
            {
                case float:
                    ForAvx<TOp>(
                        (float*)inputPtr,
                        (float*)resultPtr,
                        float.CreateChecked(scalar),
                        n);
                    return;
                case double:
                    ForAvx<TOp>(
                        (double*)inputPtr,
                        (double*)resultPtr,
                        double.CreateChecked(scalar),
                        n);
                    return;
            }
        }

        ForScalar<TOp, TNumber>(inputPtr, resultPtr, scalar, n);
    }

    private static unsafe void ForAvxFma<TOp>(float* inputPtr, float* resultPtr, float scalar, long n)
        where TOp : IUnaryOperationWithScalarAvxFma
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
                (tileIdx, _, data) =>
                {
                    InnerLoop(tileIdx, data);
                    return data;
                },
                data => NativeMemory.AlignedFree((void*)data));
        }
        else
        {
            var data = (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree((void*)data);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, nuint data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
            var count = tileEnd - tileStart;
            var dataPtr = (float*)data.ToPointer();
            var scalarBroadcast = Vector256.Create(scalar);

            NativeBufferHelpers.Pack1dFp32Avx(inputPtr + tileStart, dataPtr, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
            {
                var vector = Avx.LoadVector256(dataPtr + i);
                var result = TOp.ApplyAvxFmaFp32(vector, scalarBroadcast);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                resultPtr[tileStart + i] = TOp.ApplyTailFp32(dataPtr[i], scalar);
            }
        }
    }

    private static unsafe void ForAvxFma<TOp>(double* inputPtr, double* resultPtr, double scalar, long n)
        where TOp : IUnaryOperationWithScalarAvxFma
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
                (tileIdx, _, data) =>
                {
                    InnerLoop(tileIdx, data);
                    return data;
                },
                data => NativeMemory.AlignedFree((void*)data));
        }
        else
        {
            var data = (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree((void*)data);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, nuint data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
            var count = tileEnd - tileStart;
            var dataPtr = (double*)data.ToPointer();
            var scalarBroadcast = Vector256.Create(scalar);

            NativeBufferHelpers.Pack1dFp64Avx(inputPtr + tileStart, dataPtr, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
            {
                var vector = Avx.LoadVector256(dataPtr + i);
                var result = TOp.ApplyAvxFmaFp64(vector, scalarBroadcast);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                resultPtr[tileStart + i] = TOp.ApplyTailFp64(dataPtr[i], scalar);
            }
        }
    }

    private static unsafe void ForAvx<TOp>(float* inputPtr, float* resultPtr, float scalar, long n)
        where TOp : IUnaryOperationWithScalarAvx
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
                (tileIdx, _, data) =>
                {
                    InnerLoop(tileIdx, data);
                    return data;
                },
                data => NativeMemory.AlignedFree((void*)data));
        }
        else
        {
            var data = (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree((void*)data);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, nuint data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
            var count = tileEnd - tileStart;
            var dataPtr = (float*)data.ToPointer();
            var scalarBroadcast = Vector256.Create(scalar);

            NativeBufferHelpers.Pack1dFp32Avx(inputPtr + tileStart, dataPtr, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
            {
                var vector = Avx.LoadVector256(dataPtr + i);
                var result = TOp.ApplyAvxFp32(vector, scalarBroadcast);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                resultPtr[tileStart + i] = TOp.ApplyTailFp32(dataPtr[i], scalar);
            }
        }
    }

    private static unsafe void ForAvx<TOp>(double* inputPtr, double* resultPtr, double scalar, long n)
        where TOp : IUnaryOperationWithScalarAvx
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
                (tileIdx, _, data) =>
                {
                    InnerLoop(tileIdx, data);
                    return data;
                },
                data => NativeMemory.AlignedFree((void*)data));
        }
        else
        {
            var data = (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree((void*)data);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, nuint data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
            var count = tileEnd - tileStart;
            var scalarBroadcast = Vector256.Create(scalar);
            var dataPtr = (double*)data;

            NativeBufferHelpers.Pack1dFp64Avx(inputPtr + tileStart, dataPtr, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
            {
                var vector = Avx.LoadVector256(dataPtr + i);
                var result = TOp.ApplyAvxFp64(vector, scalarBroadcast);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                resultPtr[tileStart + i] = TOp.ApplyTailFp64(dataPtr[i], scalar);
            }
        }
    }

    private static unsafe void ForScalar<TOp, TNumber>(TNumber* inputPtr, TNumber* resultPtr, TNumber scalar, long n)
        where TOp : IUnaryOperationWithScalar<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tileSize = NativeBufferHelpers.GetTileSize<TNumber>();
        var tileCount = (n + tileSize - 1) / tileSize;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
                (tileIdx, _, data) =>
                {
                    InnerLoop(tileIdx, data);
                    return data;
                },
                data => NativeMemory.AlignedFree((void*)data));
        }
        else
        {
            var data = (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree((void*)data);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, nuint data)
        {
            var tileStart = tileIdx * tileSize;
            var tileEnd = Math.Min(tileStart + tileSize, n);
            var count = tileEnd - tileStart;
            var dataPtr = (TNumber*)data.ToPointer();

            NativeBufferHelpers.Pack1d(inputPtr + tileStart, dataPtr, count);

            for (var i = 0L; i < count; i++)
            {
                resultPtr[tileStart + i] = TOp.ApplyScalar(dataPtr[i], scalar);
            }
        }
    }
}