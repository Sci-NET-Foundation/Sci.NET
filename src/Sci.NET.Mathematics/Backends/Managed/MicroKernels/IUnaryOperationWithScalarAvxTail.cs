// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for unary operations with scalar AVX tail handling.
/// </summary>
public interface IUnaryOperationWithScalarAvxTail
{
    /// <summary>
    /// Applies the operation to the tail elements that do not fit into a full vector.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract float ApplyTailFp32(float input, float scalar);

    /// <summary>
    /// Applies the operation to the tail elements that do not fit into a full vector.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract double ApplyTailFp64(double input, double scalar);
}