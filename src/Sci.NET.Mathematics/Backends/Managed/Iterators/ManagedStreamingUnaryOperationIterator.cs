// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Backends.Managed.Buffers;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedStreamingUnaryOperationIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* inputPtr, TNumber* resultPtr, long n)
        where TOp : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        var processes = ManagedTensorBackend.GetNumThreadsByElementCount<float>(n);

        if (TOp.IsAvxFmaSupported())
        {
            switch (TNumber.Zero)
            {
                case float when processes == 1:
                    InnerLoopAvxFmaFp32<TOp>(
                        0,
                        n,
                        processes,
                        (float*)inputPtr,
                        (float*)resultPtr);
                    return;
                case float when processes > 1:
                    _ = Parallel.For(
                        0,
                        processes,
                        new ParallelOptions { MaxDegreeOfParallelism = processes },
                        tid => InnerLoopAvxFmaFp32<TOp>(
                            tid,
                            n,
                            processes,
                            (float*)inputPtr,
                            (float*)resultPtr));
                    return;
                case double when processes == 1:
                    InnerLoopAvxFmaFp64<TOp>(
                        0,
                        n,
                        processes,
                        (double*)inputPtr,
                        (double*)resultPtr);
                    return;
                case double when processes > 1:
                    _ = Parallel.For(
                        0,
                        processes,
                        new ParallelOptions { MaxDegreeOfParallelism = processes },
                        tid => InnerLoopAvxFmaFp64<TOp>(
                            tid,
                            n,
                            processes,
                            (double*)inputPtr,
                            (double*)resultPtr));
                    return;
            }
        }

        if (TOp.IsAvxSupported())
        {
            switch (TNumber.Zero)
            {
                case float when processes == 1:
                    InnerLoopAvxFp32<TOp>(
                        0,
                        (int)n,
                        processes,
                        (float*)inputPtr,
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
                            (float*)inputPtr,
                            (float*)resultPtr));
                    return;
                case double when processes == 1:
                    InnerLoopAvxFp64<TOp>(
                        0,
                        n,
                        processes,
                        (double*)inputPtr,
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
                            (double*)inputPtr,
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
                inputPtr,
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
                    inputPtr,
                    resultPtr));
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFmaFp32<TOp>(
        long tid,
        long n,
        long processes,
        float* inputPtr,
        float* resultPtr)
        where TOp : IUnaryOperationAvxFma
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(float);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
        {
            Sse.Prefetch0(inputPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var inputVector = Avx.LoadVector256(inputPtr + start + i);
            var result = TOp.ApplyAvxFmaFp32(inputVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var input = inputPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp32(input);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFmaFp64<TOp>(
        long tid,
        long n,
        long processes,
        double* inputPtr,
        double* resultPtr)
        where TOp : IUnaryOperationAvxFma
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(double);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
        {
            Sse.Prefetch0(inputPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var inputVector = Avx.LoadVector256(inputPtr + start + i);
            var result = TOp.ApplyAvxFmaFp64(inputVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var inputValue = inputPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp64(inputValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFp32<TOp>(
        long tid,
        long n,
        long processes,
        float* inputPtr,
        float* resultPtr)
        where TOp : IUnaryOperationAvx
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(float);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
        {
            Sse.Prefetch0(inputPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var inputVector = Avx.LoadVector256(inputPtr + start + i);
            var result = TOp.ApplyAvxFp32(inputVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var inputValue = inputPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp32(inputValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopAvxFp64<TOp>(
        long tid,
        long n,
        long processes,
        double* inputPtr,
        double* resultPtr)
        where TOp : IUnaryOperationAvx
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        const int prefetchDistance = 256;
        const int prefetchVectorCount = prefetchDistance / sizeof(double);

        var i = 0;
        for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
        {
            Sse.Prefetch0(inputPtr + start + i + prefetchVectorCount);
            Sse.PrefetchNonTemporal(resultPtr + start + i + prefetchVectorCount);

            var inputVector = Avx.LoadVector256(inputPtr + start + i);
            var result = TOp.ApplyAvxFp64(inputVector);

            Avx.Store(resultPtr + start + i, result);
        }

        for (; i < count; i++)
        {
            var inputValue = inputPtr[start + i];
            resultPtr[start + i] = TOp.ApplyTailFp64(inputValue);
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private static unsafe void InnerLoopScalar<TOp, TNumber>(
        long tid,
        long n,
        long processes,
        TNumber* inputPtr,
        TNumber* resultPtr)
        where TOp : IUnaryOperation<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var start = tid * n / processes;
        var end = (tid + 1) * n / processes;
        var count = end - start;

        for (var i = 0; i < count; i++)
        {
            var inputValue = inputPtr[start + i];
            resultPtr[start + i] = TOp.ApplyScalar(inputValue);
        }
    }
}