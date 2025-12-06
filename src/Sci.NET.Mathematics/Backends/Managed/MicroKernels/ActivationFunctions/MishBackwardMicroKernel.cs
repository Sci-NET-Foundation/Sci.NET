// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class MishBackwardMicroKernel<TNumber> : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
    where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>, ILogarithmicFunctions<TNumber>
{
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
    public static TNumber ApplyScalar(TNumber input)
    {
        var minusOne = TNumber.Zero - TNumber.One;
        var minusTwo = TNumber.Zero - (TNumber.One + TNumber.One);
        var two = TNumber.One + TNumber.One;

        // mish'\left(x\right)=\frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}-\frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}+\frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
        var eToTheX = TNumber.Exp(input);
        var onePlusEX = TNumber.One + eToTheX;
        var onePlusEXSquared = onePlusEX * onePlusEX;

        // \frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}
        var firstTerm = (minusOne + onePlusEXSquared) / (TNumber.One + onePlusEXSquared);

        // \frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}
        var onePlusExpXSquared = (TNumber.One + onePlusEXSquared) * (TNumber.One + onePlusEXSquared);
        var secondTerm = minusTwo * eToTheX * onePlusEX * (minusOne + onePlusEXSquared) * input / onePlusExpXSquared;

        // \frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
        var thirdTerm = two * eToTheX * onePlusEX * input / (TNumber.One + onePlusEXSquared);

        return firstTerm + secondTerm + thirdTerm;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFmaFp32(Vector256<float> input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFmaFp64(Vector256<double> input)
    {
        throw new NotSupportedException("FMA instruction set is not applicable for this operation.");
    }
}