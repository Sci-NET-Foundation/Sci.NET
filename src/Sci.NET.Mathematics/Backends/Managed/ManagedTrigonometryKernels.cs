// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Exponential;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Trigonometry;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Trigonometry.Backwards;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedTrigonometryKernels : ITrigonometryKernels
{
    public unsafe void Sin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CosMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Tan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<TanMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CosMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Tan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<TanMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinhMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CoshMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Tanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<TanhMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CoshMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Tanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<TanhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ASinMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ACosMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Atan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ATanMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ASinMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ACosMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Atan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ATanMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ASinhMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ACoshMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Atanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ATanhMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ASinhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ACoshMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Atanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ATanhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Csc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CosMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<TanMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Csc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinMicroKernel<TNumber>, SquareMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CosMicroKernel<TNumber>, SquareMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CotMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Csch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinhMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CoshMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Coth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CothMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Csch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<SinhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CoshMicroKernel<TNumber>, SquareMicroKernel<TNumber>, ReciprocalMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Coth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<CothMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acsc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ASinMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ACosMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ATanMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acsc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ASinMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ACosMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ATanMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acsch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ASinhMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ACoshMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acoth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ATanhMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acsch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ASinhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ACoshMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acoth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedUnaryOperationIterator.Apply<ReciprocalMicroKernel<TNumber>, ATanhMicroKernel<TNumber>, SquareMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            tensor.Shape.ElementCount,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void SinBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<SinBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void CosBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<CosBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void TanBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<TanBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sin2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Sin2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cos2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Cos2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Tan2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Tan2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void SinhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<SinhBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void CoshBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<CoshBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void TanhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<TanhBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sinh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Sinh2Cosh2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cosh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Sinh2Cosh2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Tanh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Tanh2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AsinBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AsinBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AcosBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AcosBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AtanBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AtanBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asin2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Asin2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acos2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Acos2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Atan2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Atan2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AsinhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AsinhBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AcoshBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AcoshBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AtanhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AtanhBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asinh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Asinh2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acosh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Acosh2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Atanh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Atanh2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void CscBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<CscBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void SecBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<SecBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void CotBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<CotBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Csc2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Csc2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sec2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Sec2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Cot2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Cot2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void CschBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<CschBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void SechBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<SechBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void CothBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<CothBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Csch2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Csch2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Sech2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Sech2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Coth2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Coth2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AcscBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AcscBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AsecBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AsecBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AcotBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AcotBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acsc2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Acsc2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asec2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Asec2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acot2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Acot2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AcschBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AcschBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AsechBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AsechBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void AcothBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<AcothBackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acsch2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Acsch2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Asech2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Asech2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void Acoth2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        ManagedBinaryOperationIterator.Apply<Acoth2BackwardsMicroKernel<TNumber>, TNumber>(
            tensor.Memory.ToPointer(),
            gradient.Memory.ToPointer(),
            result.Memory.ToPointer(),
            result.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }
}