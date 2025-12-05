// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Performance;
using Sci.NET.Mathematics.Backends.Managed.Buffers;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedBlockedBinaryOperationIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* leftPtr, TNumber* rightPtr, TNumber* resultPtr, long n)
        where TOp : IBinaryOperation<TNumber>, IBinaryOperationAvx, IBinaryOperationAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (TOp.IsAvxFmaSupported())
        {
            switch (TNumber.Zero)
            {
                case float:
                    ForAvxFma<TOp>(
                        (float*)leftPtr,
                        (float*)rightPtr,
                        (float*)resultPtr,
                        n);
                    return;
                case double:
                    ForAvxFma<TOp>(
                        (double*)leftPtr,
                        (double*)rightPtr,
                        (double*)resultPtr,
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
                        (float*)leftPtr,
                        (float*)rightPtr,
                        (float*)resultPtr,
                        n);
                    return;
                case double:
                    ForAvx<TOp>(
                        (double*)leftPtr,
                        (double*)rightPtr,
                        (double*)resultPtr,
                        n);
                    return;
            }
        }

        ForScalar<TOp, TNumber>(leftPtr, rightPtr, resultPtr, n);
    }

    private static unsafe void ForAvxFma<TOp>(float* leftPtr, float* rightPtr, float* resultPtr, long n)
        where TOp : IBinaryOperationAvxFma
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () =>
                {
                    var left = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                    var right = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                    return new Panel2dFp32(left, right);
                },
                (tileIdx, _, panels) =>
                {
                    InnerLoop(tileIdx, panels);

                    return panels;
                },
                panels =>
                {
                    NativeMemory.AlignedFree(panels.A);
                    NativeMemory.AlignedFree(panels.B);
                });
        }
        else
        {
            var leftPanel = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var rightPanel = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var data = new Panel2dFp32(leftPanel, rightPanel);

            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree(leftPanel);
            NativeMemory.AlignedFree(rightPanel);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, Panel2dFp32 data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
            var count = tileEnd - tileStart;
            var leftPanel = data.A;
            var rightPanel = data.B;

            NativeBufferHelpers.Pack1dFp32Avx(leftPtr + tileStart, leftPanel, count);
            NativeBufferHelpers.Pack1dFp32Avx(rightPtr + tileStart, rightPanel, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
            {
                var leftVector = Avx.LoadVector256(leftPanel + i);
                var rightVector = Avx.LoadVector256(rightPanel + i);

                var result = TOp.ApplyAvxFmaFp32(leftVector, rightVector);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                var leftValue = leftPanel[i];
                var rightValue = rightPanel[i];
                resultPtr[tileStart + i] = TOp.ApplyTailFp32(leftValue, rightValue);
            }
        }
    }

    private static unsafe void ForAvxFma<TOp>(double* leftPtr, double* rightPtr, double* resultPtr, long n)
        where TOp : IBinaryOperationAvxFma
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () =>
                {
                    var left = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                    var right = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                    return new Panel2dFp64(left, right);
                },
                (tileIdx, _, panels) =>
                {
                    InnerLoop(tileIdx, panels);

                    return panels;
                },
                (panels) =>
                {
                    NativeMemory.AlignedFree(panels.A);
                    NativeMemory.AlignedFree(panels.B);
                });
        }
        else
        {
            var leftPanel = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var rightPanel = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var data = new Panel2dFp64(leftPanel, rightPanel);

            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree(leftPanel);
            NativeMemory.AlignedFree(rightPanel);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, Panel2dFp64 data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
            var count = tileEnd - tileStart;
            var leftPanel = data.A;
            var rightPanel = data.B;

            NativeBufferHelpers.Pack1dFp64Avx(leftPtr + tileStart, leftPanel, count);
            NativeBufferHelpers.Pack1dFp64Avx(rightPtr + tileStart, rightPanel, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
            {
                var leftVector = Avx.LoadVector256(leftPanel + i);
                var rightVector = Avx.LoadVector256(rightPanel + i);

                var result = TOp.ApplyAvxFmaFp64(leftVector, rightVector);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                var leftValue = leftPanel[i];
                var rightValue = rightPanel[i];
                resultPtr[tileStart + i] = TOp.ApplyTailFp64(leftValue, rightValue);
            }
        }
    }

    private static unsafe void ForAvx<TOp>(float* leftPtr, float* rightPtr, float* resultPtr, long n)
        where TOp : IBinaryOperationAvx
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () =>
                {
                    var left = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                    var right = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                    return new Panel2dFp32(left, right);
                },
                (tileIdx, _, panels) =>
                {
                    InnerLoop(tileIdx, panels);

                    return panels;
                },
                panels =>
                {
                    NativeMemory.AlignedFree(panels.A);
                    NativeMemory.AlignedFree(panels.B);
                });
        }
        else
        {
            var leftPanel = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var rightPanel = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var data = new Panel2dFp32(leftPanel, rightPanel);

            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree(leftPanel);
            NativeMemory.AlignedFree(rightPanel);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, Panel2dFp32 data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
            var count = tileEnd - tileStart;
            var leftPanel = data.A;
            var rightPanel = data.B;

            NativeBufferHelpers.Pack1dFp32Avx(leftPtr + tileStart, leftPanel, count);
            NativeBufferHelpers.Pack1dFp32Avx(rightPtr + tileStart, rightPanel, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
            {
                var leftVector = Avx.LoadVector256(leftPanel + i);
                var rightVector = Avx.LoadVector256(rightPanel + i);

                var result = TOp.ApplyAvxFp32(leftVector, rightVector);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                var leftValue = leftPanel[i];
                var rightValue = rightPanel[i];
                resultPtr[tileStart + i] = TOp.ApplyTailFp32(leftValue, rightValue);
            }
        }
    }

    private static unsafe void ForAvx<TOp>(double* leftPtr, double* rightPtr, double* resultPtr, long n)
        where TOp : IBinaryOperationAvx
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () =>
                {
                    var left = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                    var right = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                    return new Panel2dFp64(left, right);
                },
                (tileIdx, _, panels) =>
                {
                    InnerLoop(tileIdx, panels);

                    return panels;
                },
                (panels) =>
                {
                    NativeMemory.AlignedFree(panels.A);
                    NativeMemory.AlignedFree(panels.B);
                });
        }
        else
        {
            var leftPanel = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var rightPanel = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var data = new Panel2dFp64(leftPanel, rightPanel);

            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree(leftPanel);
            NativeMemory.AlignedFree(rightPanel);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, Panel2dFp64 data)
        {
            var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
            var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
            var count = tileEnd - tileStart;
            var leftPanel = data.A;
            var rightPanel = data.B;

            NativeBufferHelpers.Pack1dFp64Avx(leftPtr + tileStart, leftPanel, count);
            NativeBufferHelpers.Pack1dFp64Avx(rightPtr + tileStart, rightPanel, count);

            var i = 0L;
            for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
            {
                var leftVector = Avx.LoadVector256(leftPanel + i);
                var rightVector = Avx.LoadVector256(rightPanel + i);

                var result = TOp.ApplyAvxFp64(leftVector, rightVector);

                Avx.Store(resultPtr + tileStart + i, result);
            }

            for (; i < count; i++)
            {
                var leftValue = leftPanel[i];
                var rightValue = rightPanel[i];
                resultPtr[tileStart + i] = TOp.ApplyTailFp64(leftValue, rightValue);
            }
        }
    }

    private static unsafe void ForScalar<TOp, TNumber>(TNumber* leftPtr, TNumber* rightPtr, TNumber* resultPtr, long n)
        where TOp : IBinaryOperation<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var vectorSize = NativeBufferHelpers.GetTileSize<TNumber>();
        var tileCount = (n + vectorSize - 1) / vectorSize;

        if (ManagedTensorBackend.ShouldParallelizeForTiles(tileCount))
        {
            _ = Parallel.For(
                0,
                tileCount,
                new ParallelOptions { MaxDegreeOfParallelism = ManagedTensorBackend.GetMaxDegreeOfParallelism(tileCount) },
                () =>
                {
                    var left = (TNumber*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                    var right = (TNumber*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                    return new Panel2d<TNumber>(left, right);
                },
                (tileIdx, _, panels) =>
                {
                    InnerLoop(tileIdx, panels);

                    return panels;
                },
                (panels) =>
                {
                    NativeMemory.AlignedFree(panels.A);
                    NativeMemory.AlignedFree(panels.B);
                });
        }
        else
        {
            var leftPanel = (TNumber*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var rightPanel = (TNumber*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
            var data = new Panel2d<TNumber>(leftPanel, rightPanel);

            for (var tileIdx = 0; tileIdx < tileCount; tileIdx++)
            {
                InnerLoop(tileIdx, data);
            }

            NativeMemory.AlignedFree(leftPanel);
            NativeMemory.AlignedFree(rightPanel);
        }

        [MethodImpl(ImplementationOptions.HotPath)]
        void InnerLoop(long tileIdx, Panel2d<TNumber> data)
        {
            var tileSize = NativeBufferHelpers.GetTileSize<TNumber>();
            var tileStart = tileIdx * tileSize;
            var tileEnd = Math.Min(tileStart + tileSize, n);
            var count = tileEnd - tileStart;
            var leftPanel = data.A;
            var rightPanel = data.B;

            NativeBufferHelpers.Pack1d(leftPtr + tileStart, leftPanel, count);
            NativeBufferHelpers.Pack1d(rightPtr + tileStart, rightPanel, count);

            for (var i = 0L; i < count; i++)
            {
                var leftValue = leftPanel[i];
                var rightValue = rightPanel[i];
                resultPtr[tileStart + i] = TOp.ApplyScalar(leftValue, rightValue);
            }
        }
    }
}