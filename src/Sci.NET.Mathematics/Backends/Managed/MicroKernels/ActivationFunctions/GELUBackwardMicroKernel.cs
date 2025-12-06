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
internal class GELUBackwardMicroKernel<TNumber> : IUnaryOperation<TNumber>, IUnaryOperationAvx, IUnaryOperationAvxFma
    where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>
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
        var magicNumber1 = TNumber.CreateChecked(0.3989422804014327);
        var magicNumber2 = TNumber.CreateChecked(0.134145);
        var magicNumber3 = TNumber.CreateChecked(0.044715);
        var two = TNumber.CreateChecked(2);
        var half = TNumber.One / two;
        var sqrtPiTerm = TNumber.Sqrt(two / TNumber.Pi);

        var xSquared = input * input;
        var xCubed = input * input * input;
        var xPlusMagicXCubed = input + (magicNumber3 * xCubed);

        // \sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)
        var tanhArg = sqrtPiTerm * xPlusMagicXCubed;

        // 0.5\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)\right)
        var halfTanhTerm = half * (TNumber.One + TNumber.Tanh(tanhArg));

        // \operatorname{sech}\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)
        var sechTerm = TNumber.One / TNumber.Cosh(tanhArg);

        // \operatorname{sech}^{2}\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)
        var sechSquared = sechTerm * sechTerm;

        // 0.3989422804014327x\left(1+0.134145x^{2}\right)
        var firstMagicTerm = magicNumber1 * input * (TNumber.One + (magicNumber2 * xSquared));

        return (firstMagicTerm * sechSquared) + halfTanhTerm;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static float ApplyTailFp32(float input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static double ApplyTailFp64(double input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<float> ApplyAvxFp32(Vector256<float> input)
    {
        throw new IntrinsicTypeNotImplementedException();
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    public static Vector256<double> ApplyAvxFp64(Vector256<double> input)
    {
        throw new IntrinsicTypeNotImplementedException();
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