// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal static class ManagedParameterizedUnaryOperation
{
    [SuppressMessage("Style", "IDE0010:Add missing cases", Justification = "Reviewed")]
    public static unsafe void For<TOp, TNumber>(TNumber* inputPtr, TNumber* resultPtr, TOp instance, long n)
        where TOp : IUnaryParameterizedOperation<TOp, TNumber>, IUnaryParameterizedOperationAvx<TOp>, IUnaryParameterizedOperationAvxFma<TOp>
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (ManagedTensorBackend.ShouldStream(n))
        {
            ManagedStreamingUnaryParameterizedIterator.For(inputPtr, resultPtr, instance, n);
        }
        else
        {
            ManagedBlockedUnaryParameterizedIterator.For(inputPtr, resultPtr, instance, n);
        }
    }
}