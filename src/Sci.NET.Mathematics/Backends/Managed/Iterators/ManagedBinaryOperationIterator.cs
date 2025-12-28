// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedBinaryOperationIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* leftPtr, TNumber* rightPtr, TNumber* resultPtr, long n, CpuComputeDevice device)
        where TOp : IBinaryOperation<TNumber>, IBinaryOperationAvx, IBinaryOperationAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (ManagedTensorBackend.ShouldStream(n))
        {
            ManagedStreamingBinaryOperationIterator.For<TOp, TNumber>(leftPtr, rightPtr, resultPtr, n, device);
        }
        else
        {
            ManagedBlockedBinaryOperationIterator.For<TOp, TNumber>(leftPtr, rightPtr, resultPtr, n, device);
        }
    }
}