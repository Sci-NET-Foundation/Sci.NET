// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class CeluBackwardMicroKernel<TNumber> : IUnaryParameterizedOperation<CeluBackwardMicroKernel<TNumber>, TNumber>,
    IUnaryParameterizedOperationAvx<CeluBackwardMicroKernel<TNumber>>,
    IUnaryParameterizedOperationAvxFma<CeluBackwardMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _alpha;

    public CeluBackwardMicroKernel(MicroKernelParameter<TNumber> alpha)
    {
        _alpha = alpha;
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
    public static TNumber ApplyScalar(TNumber input, CeluBackwardMicroKernel<TNumber> instance)
    {
        return TNumber.Min(TNumber.One, instance._alpha.ScalarValue * TNumber.Exp(input / instance._alpha.ScalarValue));
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input, CeluBackwardMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input, CeluBackwardMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, CeluBackwardMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, CeluBackwardMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, CeluBackwardMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, CeluBackwardMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}