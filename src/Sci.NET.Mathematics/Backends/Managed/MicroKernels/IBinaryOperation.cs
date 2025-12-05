// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for binary vectorized operations.
/// </summary>
/// <typeparam name="TNumber">The numeric type.</typeparam>
public interface IBinaryOperation<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Invokes the scalar operation on two T inputs.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract TNumber ApplyScalar(TNumber left, TNumber right);
}