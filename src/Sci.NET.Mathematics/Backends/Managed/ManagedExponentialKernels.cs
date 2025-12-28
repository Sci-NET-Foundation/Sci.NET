// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Exponential;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedExponentialKernels : IExponentialKernels
{
    public unsafe void Pow<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power, ITensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For<PowMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new PowMicroKernel<TNumber>(power.Value),
            value.Shape.ElementCount,
            (CpuComputeDevice)value.Device);
    }

    public unsafe void PowBackwards<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power, ITensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For<PowBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new PowBackwardMicroKernel<TNumber>(power.Value),
            value.Shape.ElementCount,
            (CpuComputeDevice)value.Device);
    }

    public unsafe void Square<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<SquareMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Shape.ElementCount,
            (CpuComputeDevice)value.Device);
    }

    public unsafe void Exp<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<ExpMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Shape.ElementCount,
            (CpuComputeDevice)value.Device);
    }

    public unsafe void Log<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<LogMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Shape.ElementCount,
            (CpuComputeDevice)value.Device);
    }

    public unsafe void LogBackwards<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<LogBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Shape.ElementCount,
            (CpuComputeDevice)value.Device);
    }
}