// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedUnaryOperationIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* inputPtr, TNumber* resultPtr, long n, CpuComputeDevice device)
        where TOp : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (ManagedTensorBackend.ShouldStream(n))
        {
            ManagedStreamingUnaryOperationIterator.For<TOp, TNumber>(inputPtr, resultPtr, n, device);
        }
        else
        {
            ManagedBlockedUnaryOperationIterator.For<TOp, TNumber>(inputPtr, resultPtr, n, device);
        }
    }
}