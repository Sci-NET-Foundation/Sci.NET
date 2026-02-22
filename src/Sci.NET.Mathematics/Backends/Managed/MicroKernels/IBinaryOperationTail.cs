// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for binary operations that require tail processing in vectorized implementations, providing methods for scalar application of the operation on both single and double precision floating-point numbers.
/// </summary>
public interface IBinaryOperationTail
{
    /// <summary>
    /// Applies the operation to two float inputs, used for tail processing in vectorized implementations.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract float ApplyScalarFp32(float left, float right);

    /// <summary>
    /// Applies the operation to two double inputs, used for tail processing in vectorized implementations.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">>The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract double ApplyScalarFp64(double left, double right);
}