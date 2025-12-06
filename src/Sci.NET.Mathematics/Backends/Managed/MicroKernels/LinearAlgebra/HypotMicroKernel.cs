// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Intrinsics;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.LinearAlgebra;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class HypotMicroKernel<TNumber> : IBinaryOperation<TNumber>, IBinaryOperationAvx, IBinaryOperationAvxFma
    where TNumber : unmanaged, INumber<TNumber>, IRootFunctions<TNumber>
{
    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxFmaSupported()
    {
        return IntrinsicsHelper.IsAvxFmaSupported();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxSupported()
    {
        return IntrinsicsHelper.IsAvxSupported();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static TNumber ApplyScalar(TNumber left, TNumber right)
    {
        return TNumber.Hypot(left, right);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float left, float right)
    {
        return float.Hypot(left, right);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double left, double right)
    {
        return double.Hypot(left, right);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> left, Vector256<float> right)
    {
        var leftSquared = Avx.Multiply(left, left);
        var rightSquared = Avx.Multiply(right, right);
        var sum = Avx.Add(leftSquared, rightSquared);
        return Avx.Sqrt(sum);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> left, Vector256<double> right)
    {
        var leftSquared = Avx.Multiply(left, left);
        var rightSquared = Avx.Multiply(right, right);
        var sum = Avx.Add(leftSquared, rightSquared);
        return Avx.Sqrt(sum);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> left, Vector256<float> right)
    {
        return Avx.Sqrt(Fma.MultiplyAdd(right, right, Fma.MultiplyAdd(left, left, Vector256<float>.Zero)));
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> left, Vector256<double> right)
    {
        return Avx.Sqrt(Fma.MultiplyAdd(right, right, Fma.MultiplyAdd(left, left, Vector256<double>.Zero)));
    }
}