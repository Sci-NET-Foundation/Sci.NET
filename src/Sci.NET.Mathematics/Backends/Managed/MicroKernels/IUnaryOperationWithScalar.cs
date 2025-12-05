// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for unary vectorized operations with a scalar parameter.
/// </summary>
/// <typeparam name="TNumber">The numeric type.</typeparam>
public interface IUnaryOperationWithScalar<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Invokes the scalar operation on two T inputs.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract TNumber ApplyScalar(TNumber input, TNumber scalar);
}