// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// An interface for unary operations on numbers.
/// </summary>
/// <typeparam name="TSelf">The self type.</typeparam>
/// <typeparam name="TNumber">The number type.</typeparam>
public interface IUnaryParameterizedOperation<in TSelf, TNumber>
    where TSelf : IUnaryParameterizedOperation<TSelf, TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Invokes the scalar operation on two T inputs.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <param name="instance">The operation instance.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract TNumber ApplyScalar(TNumber input, TSelf instance);
}