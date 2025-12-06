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
internal class ReLUBackwardMicroKernel<TNumber> : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
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
    public static TNumber ApplyScalar(TNumber input)
    {
        var one = TNumber.CreateChecked(1);

        return input > TNumber.Zero ? one : TNumber.Zero;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input)
    {
        return input > 0.0f ? 1.0f : 0.0f;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input)
    {
        return input > 0.0 ? 1.0 : 0.0;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input)
    {
        var zeroVector = Vector256<float>.Zero;
        var oneVector = Vector256.Create(1.0f);
        var mask = Avx.Compare(input, zeroVector, FloatComparisonMode.OrderedGreaterThanNonSignaling);
        return Avx.And(mask, oneVector);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input)
    {
        var zeroVector = Vector256<double>.Zero;
        var oneVector = Vector256.Create(1.0d);
        var mask = Avx.Compare(input, zeroVector, FloatComparisonMode.OrderedGreaterThanNonSignaling);
        return Avx.And(mask, oneVector);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}