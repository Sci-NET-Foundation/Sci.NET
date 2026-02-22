// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
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

    public unsafe void Negate<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<NegateMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Abs<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<AbsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AbsGradient<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AbsBackwardMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AbsoluteDifference<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AbsoluteDifferenceMicroKernel<TNumber>, TNumber>(
            left.Memory.ToPointer(),
            right.Memory.ToPointer(),
            result.Memory.ToPointer(),
            left.Shape.ElementCount,
            (ICpuComputeDevice)left.Device);
    }

    public unsafe void Sqrt<TNumber>(
        ITensor<TNumber> tensor,
        ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SqrtMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }
}