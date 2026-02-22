// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sci.NET.Mathematics.Exceptions;
using Sci.NET.Mathematics.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels.ActivationFunctions;

[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "By design")]
internal class MishBackwardMicroKernel<TNumber> : IUnaryOperation<TNumber>, IUnaryOperationAvx2
    where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>, ILogarithmicFunctions<TNumber>
{
    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool HasAvx2Implementation()
    {
        return false;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static TNumber ApplyScalar(TNumber input)
    {
        var minusOne = TNumber.CreateChecked(-1);
        var minusTwo = TNumber.Zero - (TNumber.One + TNumber.One);
        var two = TNumber.CreateChecked(2);

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
    public static float ApplyScalarFp32(float input)
    {
        // mish'\left(x\right)=\frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}-\frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}+\frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
        var eToTheX = MathF.Exp(input);
        var onePlusEX = 1.0f + eToTheX;
        var onePlusEXSquared = onePlusEX * onePlusEX;

        // \frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}
        var firstTerm = (-1.0f + onePlusEXSquared) / (1.0f + onePlusEXSquared);

        // \frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}
        var onePlusExpXSquared = (1.0f + onePlusEXSquared) * (1.0f + onePlusEXSquared);
        var secondTerm = -2.0f * eToTheX * onePlusEX * (-1.0f + onePlusEXSquared) * input / onePlusExpXSquared;

        // \frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
        var thirdTerm = 2.0f * eToTheX * onePlusEX * input / (1.0f + onePlusEXSquared);

        return firstTerm + secondTerm + thirdTerm;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyScalarFp64(double input)
    {
        // mish'\left(x\right)=\frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}-\frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}+\frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
        var eToTheX = Math.Exp(input);
        var onePlusEX = 1.0 + eToTheX;
        var onePlusEXSquared = onePlusEX * onePlusEX;

        // \frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}
        var firstTerm = (-1.0 + onePlusEXSquared) / (1.0 + onePlusEXSquared);

        // \frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}
        var onePlusExpXSquared = (1.0 + onePlusEXSquared) * (1.0 + onePlusEXSquared);
        var secondTerm = -2.0 * eToTheX * onePlusEX * (-1.0 + onePlusEXSquared) * input / onePlusExpXSquared;

        // \frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
        var thirdTerm = 2.0 * eToTheX * onePlusEX * input / (1.0 + onePlusEXSquared);

        return firstTerm + secondTerm + thirdTerm;
    }

    [ExcludeFromCodeCoverage]
    public static Vector256<float> ApplyAvx2Fp32(Vector256<float> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [ExcludeFromCodeCoverage]
    public static Vector256<double> ApplyAvx2Fp64(Vector256<double> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }
}