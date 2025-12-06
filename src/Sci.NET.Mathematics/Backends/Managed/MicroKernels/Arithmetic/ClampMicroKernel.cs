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

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class ClampMicroKernel<TNumber> : IUnaryParameterizedOperation<ClampMicroKernel<TNumber>, TNumber>,
    IUnaryParameterizedOperationAvx<ClampMicroKernel<TNumber>>,
    IUnaryParameterizedOperationAvxFma<ClampMicroKernel<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly MicroKernelParameter<TNumber> _min;
    private readonly MicroKernelParameter<TNumber> _max;

    public ClampMicroKernel(MicroKernelParameter<TNumber> min, MicroKernelParameter<TNumber> max)
    {
        _min = min;
        _max = max;
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
    public static TNumber ApplyScalar(TNumber input, ClampMicroKernel<TNumber> instance)
    {
        if (input < instance._min.ScalarValue)
        {
            return instance._min.ScalarValue;
        }
        else if (input > instance._max.ScalarValue)
        {
            return instance._max.ScalarValue;
        }
        else
        {
            return input;
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input, ClampMicroKernel<TNumber> instance)
    {
        if (input < instance._min.ScalarFp32Value)
        {
            return instance._min.ScalarFp32Value;
        }
        else if (input > instance._max.ScalarFp32Value)
        {
            return instance._max.ScalarFp32Value;
        }
        else
        {
            return input;
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input, ClampMicroKernel<TNumber> instance)
    {
        if (input < instance._min.ScalarFp64Value)
        {
            return instance._min.ScalarFp64Value;
        }
        else if (input > instance._max.ScalarFp64Value)
        {
            return instance._max.ScalarFp64Value;
        }
        else
        {
            return input;
        }
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input, ClampMicroKernel<TNumber> instance)
    {
        return Avx.Min(Avx.Max(input, instance._min.Vector256ValueFp32), instance._max.Vector256ValueFp32);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input, ClampMicroKernel<TNumber> instance)
    {
        return Avx.Min(Avx.Max(input, instance._min.Vector256ValueFp64), instance._max.Vector256ValueFp64);
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, ClampMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, ClampMicroKernel<TNumber> instance)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}