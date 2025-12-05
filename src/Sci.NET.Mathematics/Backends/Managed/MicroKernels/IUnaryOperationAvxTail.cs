// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for AVX tail operations.
/// </summary>
public interface IUnaryOperationAvxTail
{
    /// <summary>
    /// Applies the operation to the tail elements that do not fit into a full vector.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract float ApplyTailFp32(float input);

    /// <summary>
    /// Applies the operation to the tail elements that do not fit into a full vector.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract double ApplyTailFp64(double input);
}