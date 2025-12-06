// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.Exponential;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class PowMicroKernel<TNumber> : IUnaryParameterizedOperation<PowMicroKernel<TNumber>, TNumber>, IUnaryParameterizedOperationAvx<PowMicroKernel<TNumber>>, IUnaryParameterizedOperationAvxFma<PowMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>, IPowerFunctions<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _exponent;

    public PowMicroKernel(MicroKernelParameter<TNumber> exponent)
    {
        _exponent = exponent;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxSupported()
    {
        return false;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsAvxFmaSupported()
    {
        return false;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static TNumber ApplyScalar(TNumber input, PowMicroKernel<TNumber> instance)
    {
        return TNumber.Pow(input, instance._exponent.ScalarValue);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input, PowMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input, PowMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, PowMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, PowMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, PowMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, PowMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}