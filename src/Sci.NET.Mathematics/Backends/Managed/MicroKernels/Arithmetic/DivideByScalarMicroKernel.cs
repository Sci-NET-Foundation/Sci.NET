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
internal class DivideByScalarMicroKernel<TNumber> : IUnaryParameterizedOperation<DivideByScalarMicroKernel<TNumber>, TNumber>,
    IUnaryParameterizedOperationAvx<DivideByScalarMicroKernel<TNumber>>,
    IUnaryParameterizedOperationAvxFma<DivideByScalarMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _divisor;

    public DivideByScalarMicroKernel(MicroKernelParameter<TNumber> divisor)
    {
        _divisor = divisor;
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
    public static TNumber ApplyScalar(TNumber input, DivideByScalarMicroKernel<TNumber> instance)
    {
        return input / instance._divisor.ScalarValue;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input, DivideByScalarMicroKernel<TNumber> instance)
    {
        return input / instance._divisor.ScalarFp32Value;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input, DivideByScalarMicroKernel<TNumber> instance)
    {
        return input / instance._divisor.ScalarFp64Value;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, DivideByScalarMicroKernel<TNumber> instance)
    {
        return Avx.Divide(input, instance._divisor.Vector256ValueFp32);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, DivideByScalarMicroKernel<TNumber> instance)
    {
        return Avx.Divide(input, instance._divisor.Vector256ValueFp64);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, DivideByScalarMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, DivideByScalarMicroKernel<TNumber> instance)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}