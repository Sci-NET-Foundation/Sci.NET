// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedArithmeticKernels : IArithmeticKernels
{
    public void Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryArithmeticOperationIterator<AddMicroKernel<TNumber>, TNumber>(left, right, result);
        iterator.Apply();
    }

    public void Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryArithmeticOperationIterator<SubtractMicroKernel<TNumber>, TNumber>(left, right, result);
        iterator.Apply();
    }

    public void Multiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryArithmeticOperationIterator<MultiplyMicroKernel<TNumber>, TNumber>(left, right, result);
        iterator.Apply();
    }

    public void Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryArithmeticOperationIterator<DivideMicroKernel<TNumber>, TNumber>(left, right, result);
        iterator.Apply();
    }

    public unsafe void Negate<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<NegateMicroKernel<TNumber>, TNumber>(
            tensor.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void Abs<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<AbsMicroKernel<TNumber>, TNumber>(
            tensor.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void AbsGradient<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> gradient, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<AbsBackwardMicroKernel<TNumber>, TNumber>(
            tensor.ToPointer(),
            gradient.ToPointer(),
            result.ToPointer(),
            n);
    }

    public unsafe void AbsoluteDifference<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.For<AbsoluteDifferenceMicroKernel<TNumber>, TNumber>(
            left.ToPointer(),
            right.ToPointer(),
            result.ToPointer(),
            left.Length);
    }

    public unsafe void Sqrt<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<SqrtMicroKernel<TNumber>, TNumber>(
            tensor.ToPointer(),
            result.ToPointer(),
            n);
    }
}