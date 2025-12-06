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
internal class HardSigmoidMicroKernel<TNumber> : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
    where TNumber : unmanaged, INumber<TNumber>
{
    public static bool IsAvxSupported()
    {
        return IntrinsicsHelper.IsAvxSupported();
    }

    public static bool IsAvxFmaSupported()
    {
        return false;
    }

    public static TNumber ApplyScalar(TNumber input)
    {
        var three = TNumber.One + TNumber.One + TNumber.One;
        var minusThree = TNumber.Zero - three;
        var six = three + three;
        var two = TNumber.One + TNumber.One;
        var half = TNumber.One / two;

        if (input <= minusThree)
        {
            return TNumber.Zero;
        }
        else if (input >= three)
        {
            return TNumber.One;
        }
        else
        {
            return (input / six) + half;
        }
    }

    public static float ApplyTailFp32(float input)
    {
        if (input <= -3.0f)
        {
            return 0.0f;
        }
        else if (input >= 3.0f)
        {
            return 1.0f;
        }
        else
        {
            return (input / 6.0f) + 0.5f;
        }
    }

    public static double ApplyTailFp64(double input)
    {
        if (input <= -3.0f)
        {
            return 0.0f;
        }
        else if (input >= 3.0f)
        {
            return 1.0f;
        }
        else
        {
            return (input / 6.0f) + 0.5f;
        }
    }

    public static Vector256<float> ApplyAvxFp32(Vector256<float> input)
    {
        var negThree = Vector256.Create(-3.0f);
        var posThree = Vector256.Create(3.0f);
        var sixth = Vector256.Create(1.0f / 6.0f);
        var half = Vector256.Create(0.5f);
        var zero = Vector256<float>.Zero;
        var one = Vector256.Create(1.0f);
        var linear = Avx.Add(Avx.Multiply(input, sixth), half);
        var maskLow = Avx.Compare(input, negThree, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling);
        var maskHigh = Avx.Compare(input, posThree, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling);
        var result = Avx.BlendVariable(linear, zero, maskLow);

        return Avx.BlendVariable(result, one, maskHigh);
    }

    public static Vector256<double> ApplyAvxFp64(Vector256<double> input)
    {
        var negThree = Vector256.Create(-3.0d);
        var posThree = Vector256.Create(3.0d);
        var sixth = Vector256.Create(1.0d / 6.0d);
        var half = Vector256.Create(0.5d);
        var zero = Vector256<double>.Zero;
        var one = Vector256.Create(1.0d);
        var linear = Avx.Add(Avx.Multiply(input, sixth), half);
        var maskLow = Avx.Compare(input, negThree, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling);
        var maskHigh = Avx.Compare(input, posThree, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling);
        var result = Avx.BlendVariable(linear, zero, maskLow);

        return Avx.BlendVariable(result, one, maskHigh);
    }

    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}