// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for unary operations that only have scalar implementations, used for tail processing in vectorized operations.
/// </summary>
public interface IUnaryOperationTail
{
    /// <summary>
    /// Applies the operation to a float input.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract float ApplyScalarFp32(float input);

    /// <summary>
    /// Applies the operation to a double input.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract double ApplyScalarFp64(double input);
}