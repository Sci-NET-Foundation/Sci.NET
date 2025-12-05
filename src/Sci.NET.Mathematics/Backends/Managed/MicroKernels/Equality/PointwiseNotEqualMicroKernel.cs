// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Intrinsics;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.Equality;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class PointwiseNotEqualMicroKernel<TNumber> : IBinaryOperation<TNumber>, IBinaryOperationAvx, IBinaryOperationAvxFma
    where TNumber : unmanaged, INumber<TNumber>
{
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
    public static TNumber ApplyScalar(TNumber left, TNumber right)
    {
        return left != right ? TNumber.One : TNumber.Zero;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float left, float right)
    {
        return left != right ? 1.0f : 0.0f;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double left, double right)
    {
        return left != right ? 1.0d : 0.0d;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> left, Vector256<float> right)
    {
        var zero = Vector256<float>.Zero;
        var one = Vector256.Create(1.0f);

        var equalsMask = Avx.CompareNotEqual(left, right);

        return Avx.BlendVariable(zero, one, equalsMask);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> left, Vector256<double> right)
    {
        var zero = Vector256<double>.Zero;
        var one = Vector256.Create(1.0d);

        var equalsMask = Avx.CompareNotEqual(left, right);

        return Avx.BlendVariable(zero, one, equalsMask);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> left, Vector256<float> right)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> left, Vector256<double> right)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}