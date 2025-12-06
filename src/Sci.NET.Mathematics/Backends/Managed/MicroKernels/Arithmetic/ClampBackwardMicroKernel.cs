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

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class ClampBackwardMicroKernel<TNumber> : IUnaryParameterizedOperation<ClampBackwardMicroKernel<TNumber>, TNumber>,
    IUnaryParameterizedOperationAvx<ClampBackwardMicroKernel<TNumber>>,
    IUnaryParameterizedOperationAvxFma<ClampBackwardMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _min;
    private readonly MicroKernelParameter<TNumber> _max;

    public ClampBackwardMicroKernel(MicroKernelParameter<TNumber> min, MicroKernelParameter<TNumber> max)
    {
        _min = min;
        _max = max;
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
    public static TNumber ApplyScalar(TNumber input, ClampBackwardMicroKernel<TNumber> instance)
    {
        if (input <= instance._min.ScalarValue || input >= instance._max.ScalarValue)
        {
            return TNumber.Zero;
        }

        return TNumber.One;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input, ClampBackwardMicroKernel<TNumber> instance)
    {
        if (input <= instance._min.ScalarFp32Value || input >= instance._max.ScalarFp32Value)
        {
            return 0.0f;
        }

        return 1.0f;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input, ClampBackwardMicroKernel<TNumber> instance)
    {
        if (input <= instance._min.ScalarFp64Value || input >= instance._max.ScalarFp64Value)
        {
            return 0.0f;
        }

        return 1.0f;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, ClampBackwardMicroKernel<TNumber> instance)
    {
        var gtMin = Avx.Compare(input, instance._min.Vector256ValueFp32, FloatComparisonMode.OrderedGreaterThanNonSignaling);
        var ltMax = Avx.Compare(input, instance._max.Vector256ValueFp32, FloatComparisonMode.OrderedLessThanNonSignaling);
        var mask = Avx.And(gtMin, ltMax);

        return Avx.BlendVariable(Vector256<float>.Zero, Vector256<float>.One, mask);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, ClampBackwardMicroKernel<TNumber> instance)
    {
        var gtMin = Avx.Compare(input, instance._min.Vector256ValueFp64, FloatComparisonMode.OrderedGreaterThanNonSignaling);
        var ltMax = Avx.Compare(input, instance._max.Vector256ValueFp64, FloatComparisonMode.OrderedLessThanNonSignaling);
        var mask = Avx.And(gtMin, ltMax);

        return Avx.BlendVariable(Vector256<double>.Zero, Vector256<double>.One, mask);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, ClampBackwardMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, ClampBackwardMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}