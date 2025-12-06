// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Exceptions;
using Sci.NET.Mathematics.Intrinsics;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class LeakyReLUMicroKernel<TNumber> : IUnaryParameterizedOperation<LeakyReLUMicroKernel<TNumber>, TNumber>,
    IUnaryParameterizedOperationAvx<LeakyReLUMicroKernel<TNumber>>,
    IUnaryParameterizedOperationAvxFma<LeakyReLUMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _alpha;

    public LeakyReLUMicroKernel(MicroKernelParameter<TNumber> alpha)
    {
        _alpha = alpha;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxSupported()
    {
        return IntrinsicsHelper.IsAvxSupported();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxFmaSupported()
    {
        return false;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static TNumber ApplyScalar(TNumber input, LeakyReLUMicroKernel<TNumber> instance)
    {
        return input > TNumber.Zero ? input : instance._alpha.ScalarValue * input;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input, LeakyReLUMicroKernel<TNumber> instance)
    {
        return input > 0.0f ? input : instance._alpha.ScalarFp32Value * input;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input, LeakyReLUMicroKernel<TNumber> instance)
    {
        return input > 0.0d ? input : instance._alpha.ScalarFp64Value * input;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, LeakyReLUMicroKernel<TNumber> instance)
    {
        var zero = Vector256<float>.Zero;
        var mask = Avx.Compare(input, zero, FloatComparisonMode.OrderedGreaterThanSignaling);
        var scaled = Avx.Multiply(input, instance._alpha.Vector256ValueFp32);

        return Avx.BlendVariable(scaled, input, mask);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, LeakyReLUMicroKernel<TNumber> instance)
    {
        var zero = Vector256<double>.Zero;
        var mask = Avx.Compare(input, zero, FloatComparisonMode.OrderedGreaterThanSignaling);
        var scaled = Avx.Multiply(input, instance._alpha.Vector256ValueFp64);

        return Avx.BlendVariable(scaled, input, mask);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, LeakyReLUMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, LeakyReLUMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}