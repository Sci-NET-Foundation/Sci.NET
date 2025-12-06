// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Exceptions;
using Sci.NET.Mathematics.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class LeakyReLUBackwardMicroKernel<TNumber> : IUnaryParameterizedOperation<LeakyReLUBackwardMicroKernel<TNumber>, TNumber>,
    IUnaryParameterizedOperationAvx<LeakyReLUBackwardMicroKernel<TNumber>>,
    IUnaryParameterizedOperationAvxFma<LeakyReLUBackwardMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _alpha;

    public LeakyReLUBackwardMicroKernel(MicroKernelParameter<TNumber> alpha)
    {
        _alpha = alpha;
    }

    public static bool IsAvxSupported()
    {
        return IntrinsicsHelper.IsAvxSupported();
    }

    public static bool IsAvxFmaSupported()
    {
        return false;
    }

    public static TNumber ApplyScalar(TNumber input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        return input > TNumber.Zero ? TNumber.One : instance._alpha.ScalarValue;
    }

    public static float ApplyTailFp32(float input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        return input > 0.0f ? 1.0f : instance._alpha.ScalarFp32Value;
    }

    public static double ApplyTailFp64(double input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        return input > 0.0d ? 1.0d : instance._alpha.ScalarFp64Value;
    }

    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        var zero = Vector256<float>.Zero;
        var one = Vector256<float>.One;
        var mask = Avx.Compare(input, zero, FloatComparisonMode.OrderedGreaterThanSignaling);

        return Avx.BlendVariable(instance._alpha.Vector256ValueFp32, one, mask);
    }

    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        var zero = Vector256<double>.Zero;
        var one = Vector256<double>.One;
        var mask = Avx.Compare(input, zero, FloatComparisonMode.OrderedGreaterThanSignaling);

        return Avx.BlendVariable(instance._alpha.Vector256ValueFp64, one, mask);
    }

    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, LeakyReLUBackwardMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}