// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Equality;
using Sci.NET.Mathematics.Memory;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedEqualityOperationKernels : IEqualityOperationKernels
{
    public unsafe void PointwiseEqual<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBlockedBinaryOperationIterator.For<PointwiseEqualMicroKernel<TNumber>, TNumber>(
            leftOperand.ToPointer(),
            rightOperand.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void PointwiseNotEqual<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBlockedBinaryOperationIterator.For<PointwiseNotEqualMicroKernel<TNumber>, TNumber>(
            leftOperand.ToPointer(),
            rightOperand.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void PointwiseGreaterThan<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseGreaterThanMicroKernel<TNumber>, TNumber>(
            leftOperand.ToPointer(),
            rightOperand.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void PointwiseGreaterThanOrEqual<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseGreaterThanOrEqualMicroKernel<TNumber>, TNumber>(
            leftOperand.ToPointer(),
            rightOperand.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void PointwiseLessThan<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseLessThanMicroKernel<TNumber>, TNumber>(
            leftOperand.ToPointer(),
            rightOperand.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void PointwiseLessThanOrEqual<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<PointwiseLessThanOrEqualMicroKernel<TNumber>, TNumber>(
            leftOperand.ToPointer(),
            rightOperand.ToPointer(),
            result.ToPointer(),
            n);
    }
}