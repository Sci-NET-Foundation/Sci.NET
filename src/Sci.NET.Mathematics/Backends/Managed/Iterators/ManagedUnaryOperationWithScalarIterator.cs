// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedUnaryOperationWithScalarIterator
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* inputPtr, TNumber* resultPtr, TNumber scalar, long n)
        where TOp : IUnaryOperationWithScalar<TNumber>, IUnaryOperationWithScalarAvx, IUnaryOperationWithScalarAvxFma
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (ManagedTensorBackend.ShouldStream(n))
        {
            ManagedStreamingUnaryOperationWithScalarIterator.For<TOp, TNumber>(inputPtr, resultPtr, scalar, n);
        }
        else
        {
            ManagedBlockedUnaryOperationWithScalarIterator.For<TOp, TNumber>(inputPtr, resultPtr, scalar, n);
        }
    }
}