// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Exponential;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedActivationFunctionKernels : IActivationFunctionKernels
{
    public unsafe void Sigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SigmoidMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SigmoidBackard<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SigmoidBackardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void ReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReLUMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void ReLUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReLUBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void Softmax<TNumber>(ITensor<TNumber> value, Scalar<TNumber> sumBuffer, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ExpMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);

        sumBuffer.Backend.Reduction.ReduceAdd(result, Enumerable.Range(0, value.Shape.Rank).ToArray(), sumBuffer);

        ManagedUnaryOperationIterator.Apply(
            result.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new DivideByScalarMicroKernel<TNumber>(sumBuffer.Value),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SoftmaxBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> softmaxValue, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SoftmaxMicroKernel<TNumber>, TNumber>(
            softmaxValue.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void LeakyReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new LeakyReLUMicroKernel<TNumber>(alpha),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void LeakyReLUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new LeakyReLUBackwardMicroKernel<TNumber>(alpha),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void Elu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new EluMicroKernel<TNumber>(alpha),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void EluBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new EluBackwardMicroKernel<TNumber>(alpha),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void Celu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new CeluMicroKernel<TNumber>(alpha),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void CeluBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new CeluBackwardMicroKernel<TNumber>(alpha),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void Swish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SwishMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SwishBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SwishBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void Mish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<MishMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void MishBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<MishBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void HardTanh<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new ClampMicroKernel<TNumber>(min, max),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void HardTanhBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new ClampBackwardMicroKernel<TNumber>(min, max),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void HardSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<HardSigmoidMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void HardSigmoidBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<HardSigmoidBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void LogSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<LogSigmoidMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void LogSigmoidBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<LogSigmoidBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void GELU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<GELUMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void GELUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<GELUBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SoftPlus<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SoftPlusMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SoftPlusBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SoftPlusBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SoftSign<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SoftSignMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }

    public unsafe void SoftSignBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SoftSignBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length,
            (ICpuComputeDevice)value.Device);
    }
}