// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Equality;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedEqualityOperationKernels : IEqualityOperationKernels
{
    public unsafe void PointwiseEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBlockedBinaryOperationIterator.For<PointwiseEqualMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (CpuComputeDevice)left.Device);
    }

    public unsafe void PointwiseNotEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBlockedBinaryOperationIterator.For<PointwiseNotEqualMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (CpuComputeDevice)left.Device);
    }

    public unsafe void PointwiseGreaterThan<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseGreaterThanMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (CpuComputeDevice)left.Device);
    }

    public unsafe void PointwiseGreaterThanOrEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseGreaterThanOrEqualMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (CpuComputeDevice)left.Device);
    }

    public unsafe void PointwiseLessThan<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseLessThanMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (CpuComputeDevice)left.Device);
    }

    public unsafe void PointwiseLessThanOrEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseLessThanOrEqualMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (CpuComputeDevice)left.Device);
    }
}