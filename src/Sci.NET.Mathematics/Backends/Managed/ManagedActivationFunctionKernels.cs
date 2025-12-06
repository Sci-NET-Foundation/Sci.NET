// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Exponential;
using Sci.NET.Mathematics.Concurrency;
using Sci.NET.Mathematics.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedActivationFunctionKernels : IActivationFunctionKernels
{
    public unsafe void Sigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SigmoidMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void SigmoidBackard<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SigmoidBackardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void ReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<ReLUMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void ReLUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<ReLUBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void Softmax<TNumber>(ITensor<TNumber> value, Scalar<TNumber> sumBuffer, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<ExpMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);

        sumBuffer.Backend.Reduction.ReduceAddAll(result, sumBuffer);

        ManagedParameterizedUnaryOperation.For(
            result.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new DivideByScalarMicroKernel<TNumber>(sumBuffer.Value),
            result.Memory.Length);
    }

    public unsafe void SoftmaxBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> softmaxValue, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SoftmaxMicroKernel<TNumber>, TNumber>(
            softmaxValue.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void LeakyReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new LeakyReLUMicroKernel<TNumber>(alpha),
            value.Memory.Length);
    }

    public unsafe void LeakyReLUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new LeakyReLUBackwardMicroKernel<TNumber>(alpha),
            value.Memory.Length);
    }

    public unsafe void Elu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new EluMicroKernel<TNumber>(alpha),
            value.Memory.Length);
    }

    public unsafe void EluBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new EluBackwardMicroKernel<TNumber>(alpha),
            value.Memory.Length);
    }

    public unsafe void Celu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new CeluMicroKernel<TNumber>(alpha),
            value.Memory.Length);
    }

    public unsafe void CeluBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        ManagedParameterizedUnaryOperation.For(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new CeluBackwardMicroKernel<TNumber>(alpha),
            value.Memory.Length);
    }

    public unsafe void Swish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SwishMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void SwishBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SwishBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void Mish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<MishMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void MishBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<MishBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public void HardTanh<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
#pragma warning disable IDE0045
                if (inputMemory[i] < min)
                {
                    outputMemory[i] = min;
                }
                else if (inputMemory[i] > max)
                {
                    outputMemory[i] = max;
                }
                else
                {
                    outputMemory[i] = inputMemory[i];
                }
#pragma warning restore IDE0045
            });
    }

    public void HardTanhBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
#pragma warning disable IDE0045
                if (inputMemory[i] <= min || inputMemory[i] >= max)
                {
                    outputMemory[i] = TNumber.Zero;
                }
                else
                {
                    outputMemory[i] = TNumber.One;
                }
#pragma warning restore IDE0045
            });
    }

    public unsafe void HardSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<HardSigmoidMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void HardSigmoidBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<HardSigmoidBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void LogSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<LogSigmoidMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void LogSigmoidBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<LogSigmoidBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void GELU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<GELUMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void GELUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<GELUBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void SoftPlus<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SoftPlusMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void SoftPlusBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.For<SoftPlusBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void SoftSign<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<SoftSignMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }

    public unsafe void SoftSignBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.For<SoftSignBackwardMicroKernel<TNumber>, TNumber>(
            value.Memory.ToPointer(),
            result.Memory.ToPointer(),
            value.Memory.Length);
    }
}