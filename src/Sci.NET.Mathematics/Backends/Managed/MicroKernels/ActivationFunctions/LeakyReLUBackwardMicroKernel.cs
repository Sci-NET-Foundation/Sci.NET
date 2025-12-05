// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class LeakyReLUBackwardMicroKernel<TNumber> : IUnaryOperationWithScalar<TNumber>, IUnaryOperationWithScalarAvx, IUnaryOperationWithScalarAvxFma
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

    public static TNumber ApplyScalar(TNumber input, TNumber scalar)
    {
        return input > TNumber.Zero ? TNumber.One : scalar;
    }

    public static float ApplyTailFp32(float input, float scalar)
    {
        return input > 0.0f ? 1.0f : scalar;
    }

    public static double ApplyTailFp64(double input, double scalar)
    {
        return input > 0.0d ? 1.0d : scalar;
    }

    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, Vector256<float> scalar)
    {
        var zero = Vector256<float>.Zero;
        var one = Vector256<float>.One;
        var mask = Avx.Compare(input, zero, FloatComparisonMode.OrderedGreaterThanSignaling);

        return Avx.BlendVariable(scalar, one, mask);
    }

    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, Vector256<double> scalar)
    {
        var zero = Vector256<double>.Zero;
        var one = Vector256<double>.One;
        var mask = Avx.Compare(input, zero, FloatComparisonMode.OrderedGreaterThanSignaling);

        return Avx.BlendVariable(scalar, one, mask);
    }

    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, Vector256<float> scalar)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, Vector256<double> scalar)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}