// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class HardSigmoidBackwardMicroKernel<TNumber> : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
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
        var zero = TNumber.Zero;

        if (input < minusThree || input > three)
        {
            return zero;
        }

        return TNumber.One / six;
    }

    public static float ApplyTailFp32(float input)
    {
        if (input is < -3.0f or > 3.0f)
        {
            return 0.0f;
        }

        return 1.0f / 6.0f;
    }

    public static double ApplyTailFp64(double input)
    {
        if (input is < -3.0d or > 3.0d)
        {
            return 0.0d;
        }

        return 1.0d / 6.0d;
    }

    public static Vector256<float> ApplyAvxFp32(Vector256<float> input)
    {
        var negThree = Vector256.Create(-3.0f);
        var posThree = Vector256.Create(3.0f);
        var oneSixth = Vector256.Create(1.0f / 6.0f);

        var gtNegThree = Avx.Compare(input, negThree, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
        var ltPosThree = Avx.Compare(input, posThree, FloatComparisonMode.OrderedLessThanOrEqualSignaling);

        var inRange = Avx.And(gtNegThree, ltPosThree);

        return Avx.And(inRange, oneSixth);
    }

    public static Vector256<double> ApplyAvxFp64(Vector256<double> input)
    {
        var negThree = Vector256.Create(-3.0d);
        var posThree = Vector256.Create(3.0d);
        var oneSixth = Vector256.Create(1.0d / 6.0d);

        var gtNegThree = Avx.Compare(input, negThree, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
        var ltPosThree = Avx.Compare(input, posThree, FloatComparisonMode.OrderedLessThanOrEqualSignaling);

        var inRange = Avx.And(gtNegThree, ltPosThree);

        return Avx.And(inRange, oneSixth);
    }

    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}