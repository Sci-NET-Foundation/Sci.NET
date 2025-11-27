// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class NormService : INormService
{
    public Scalar<TNumber> VectorNorm<TNumber>(Vector<TNumber> vector, Scalar<TNumber>? order = null)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        using var abs = vector.Abs();
        var scalarOrder = order?.Value;

        order?.To(vector.Device);

        if (scalarOrder is null || scalarOrder == TNumber.CreateChecked(2))
        {
            using var squared = vector.Square();
            using var sum = squared.Sum().ToScalar();

            return sum.Sqrt();
        }

        if (scalarOrder == TNumber.CreateChecked(1))
        {
            return abs.Sum().ToScalar();
        }

        if (TNumber.IsPositiveInfinity((TNumber)scalarOrder))
        {
            return abs.Max().ToScalar();
        }

        if (TNumber.IsNegativeInfinity((TNumber)scalarOrder))
        {
            return abs.Min().ToScalar();
        }

        if (scalarOrder == TNumber.Zero)
        {
            throw new NotSupportedException("Zero-norm is not supported.");
        }

        using var power = new Scalar<TNumber>((TNumber)scalarOrder, backend: vector.Backend);
        using var inversePower = new Scalar<TNumber>(TNumber.One / (TNumber)scalarOrder, backend: vector.Backend);
        using var sumVectorToPower = abs.Pow(power);
        using var sumVectorToPowerSum = sumVectorToPower.Sum();

        return sumVectorToPowerSum.Pow(inversePower).ToScalar();
    }
}