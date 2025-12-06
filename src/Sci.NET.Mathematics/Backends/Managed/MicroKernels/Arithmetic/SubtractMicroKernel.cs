// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Mathematics.Intrinsics;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class SubtractMicroKernel<TNumber> : IBinaryOperation<TNumber>, IBinaryOperationAvx
    where TNumber : unmanaged, INumber<TNumber>
{
    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxSupported()
    {
        return IntrinsicsHelper.IsAvxSupported();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static TNumber ApplyScalar(TNumber left, TNumber right)
    {
        return left - right;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float left, float right)
    {
        return left - right;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double left, double right)
    {
        return left - right;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> left, Vector256<float> right)
    {
        return Avx.Subtract(left, right);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> left, Vector256<double> right)
    {
        return Avx.Subtract(left, right);
    }
}