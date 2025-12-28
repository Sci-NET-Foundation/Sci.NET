// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.Buffers;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedStreamingBinaryOperationIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* leftPtr, TNumber* rightPtr, TNumber* resultPtr, long n, CpuComputeDevice device)
        where TOp : IBinaryOperation<TNumber>, IBinaryOperationAvx, IBinaryOperationAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        var processes = ManagedTensorBackend.GetNumThreadsByElementCount<float>(n);

        if (device.IsAvxFmaSupported() && TOp.IsAvxFmaSupported())
        {
            switch (TNumber.Zero)
            {
                case float when processes == 1:
                    InnerLoopAvxFma<TOp>(
                        0,
                        n,
                        processes,
                        (float*)leftPtr,
                        (float*)rightPtr,
                        (float*)resultPtr);
                    return;
                case float when processes > 1:
                    _ = Parallel.For(
                        0,
                        processes,
                        new ParallelOptions { MaxDegreeOfParallelism = processes },
                        tid => InnerLoopAvxFma<TOp>(
                            tid,
                            n,
                            processes,
                            (float*)leftPtr,
                            (float*)rightPtr,
                            (float*)resultPtr));
                    return;
                case double when processes == 1:
                    InnerLoopAvxFma<TOp>(
                        0,
                        n,
                        processes,
                        (double*)leftPtr,
                        (double*)rightPtr,
                        (double*)resultPtr);
                    return;
                case double when processes > 1:
                    _ = Parallel.For(
                        0,
                        processes,
                        new ParallelOptions { MaxDegreeOfParallelism = processes },
                        tid => InnerLoopAvxFma<TOp>(
                            tid,
                            n,
                            processes,
                            (double*)leftPtr,
                            (double*)rightPtr,
                            (double*)resultPtr));
                    return;
            }
        }

        if (device.IsAvxSupported() && TOp.IsAvxSupported())
        {
            switch (TNumber.Zero)
            {
                case float when processes == 1:
                    InnerLoopAvxFp32<TOp>(
                        0,
                        (int)n,
                        processes,
                        (float*)leftPtr,
                        (float*)rightPtr,
                        (float*)resultPtr);
                    return;
                case float when processes > 1:
                    _ = Parallel.For(
                        0,
                        processes,
                        new ParallelOptions { MaxDegreeOfParallelism = processes },
                        tid => InnerLoopAvxFp32<TOp>(
                            tid,
                            (int)n,
                            processes,
                            (float*)leftPtr,
                            (float*)rightPtr,
                            (float*)resultPtr));
                    return;
                case double when processes == 1:
                    InnerLoopAvxFp64<TOp>(
                        0,
                        n,
                        processes,
                        (double*)leftPtr,
                        (double*)rightPtr,
                        (double*)resultPtr);
                    return;
                case double when processes > 1:
                    _ = Parallel.For(
                        0,
                        processes,
                        new ParallelOptions { MaxDegreeOfParallelism = processes },
                        tid => InnerLoopAvxFp64<TOp>(
                            tid,
                            n,
                            processes,
                            (double*)leftPtr,
                            (double*)rightPtr,
                            (double*)resultPtr));
                    return;
            }
        }

        if (processes == 1)
        {
            InnerLoopScalar<TOp, TNumber>(
                0,
                n,
                processes,
                leftPtr,
                rightPtr,
                resultPtr);
        }
        else
        {
            _ = Parallel.For(
                0,
                processes,
                new ParallelOptions { MaxDegreeOfParallelism = processes },
                tid => InnerLoopScalar<TOp, TNumber>(
                    tid,
                    n,
                    processes,
                    leftPtr,
                    rightPtr,
                    resultPtr));
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFp32<TOp>(
        long tid,
        long n,
        long processes,
        float* leftPtr,
        float* rightPtr,
        float* resultPtr)
        where TOp : IBinaryOperationAvx
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(float);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
        {
            Sse.Prefetch0(leftPtr + start + i + prefetchVectorCount);
            Sse.Prefetch0(rightPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var leftVector = Avx.LoadVector256(leftPtr + start + i);
            var rightVector = Avx.LoadVector256(rightPtr + start + i);
            var result = TOp.ApplyAvxFp32(leftVector, rightVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var leftValue = leftPtr[start + i];
            var rightValue = rightPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp32(leftValue, rightValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFma<TOp>(
        long tid,
        long n,
        long processes,
        float* leftPtr,
        float* rightPtr,
        float* resultPtr)
        where TOp : IBinaryOperationAvxFma
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(float);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
        {
            Sse.Prefetch0(leftPtr + start + i + prefetchVectorCount);
            Sse.Prefetch0(rightPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var leftVector = Avx.LoadVector256(leftPtr + start + i);
            var rightVector = Avx.LoadVector256(rightPtr + start + i);
            var result = TOp.ApplyAvxFmaFp32(leftVector, rightVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var leftValue = leftPtr[start + i];
            var rightValue = rightPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp32(leftValue, rightValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFp64<TOp>(
        long tid,
        long n,
        long processes,
        double* leftPtr,
        double* rightPtr,
        double* resultPtr)
        where TOp : IBinaryOperationAvx
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(double);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
        {
            Sse.Prefetch0(leftPtr + start + i + prefetchVectorCount);
            Sse.Prefetch0(rightPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var leftVector = Avx.LoadVector256(leftPtr + start + i);
            var rightVector = Avx.LoadVector256(rightPtr + start + i);
            var result = TOp.ApplyAvxFp64(leftVector, rightVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var leftValue = leftPtr[start + i];
            var rightValue = rightPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp64(leftValue, rightValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFma<TOp>(
        long tid,
        long n,
        long processes,
        double* leftPtr,
        double* rightPtr,
        double* resultPtr)
        where TOp : IBinaryOperationAvxFma
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(double);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
        {
            Sse.Prefetch0(leftPtr + start + i + prefetchVectorCount);
            Sse.Prefetch0(rightPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var leftVector = Avx.LoadVector256(leftPtr + start + i);
            var rightVector = Avx.LoadVector256(rightPtr + start + i);
            var result = TOp.ApplyAvxFmaFp64(leftVector, rightVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var leftValue = leftPtr[start + i];
            var rightValue = rightPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp64(leftValue, rightValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopScalar<TOp, TNumber>(
        long tid,
        long n,
        long processes,
        TNumber* leftPtr,
        TNumber* rightPtr,
        TNumber* resultPtr)
        where TOp : IBinaryOperation<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        for (var i = 0; i < count; i++)
        {
            var leftValue = leftPtr[start + i];
            var rightValue = rightPtr[start + i];
            resultPtr[start + i] = TOp.ApplyScalar(leftValue, rightValue);
        }
    }
}