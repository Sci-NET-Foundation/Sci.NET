// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Defines a reduction operation for use in reduction micro-kernels.
/// </summary>
/// <typeparam name="TNumber">The number type of the reduction operation.</typeparam>
public interface IReductionOperation<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the identity value for the reduction operation.
    /// </summary>
    public static abstract TNumber Identity { get; }

    /// <summary>
    /// Accumulates a value into the current accumulated value.
    /// </summary>
    /// <param name="current">The current accumulated value.</param>
    /// <param name="value">The value to accumulate.</param>
    /// <returns>The new accumulated value.</returns>
    public static abstract TNumber Accumulate(TNumber current, TNumber value);

    /// <summary>
    /// Finalizes the accumulated value after all values have been processed.
    /// </summary>
    /// <param name="accumulated">The accumulated value.</param>
    /// <param name="count">The number of values that were accumulated.</param>
    /// <returns>The finalized value.</returns>
    public static abstract TNumber Finalize(TNumber accumulated, long count);
}