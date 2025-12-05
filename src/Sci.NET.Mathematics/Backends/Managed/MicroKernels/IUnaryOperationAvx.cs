// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for AVX unary operations.
/// </summary>
public interface IUnaryOperationAvx : IUnaryOperationAvxTail
{
    /// <summary>
    /// Determines whether AVX is supported on the current machine.
    /// </summary>
    /// <returns>A boolean indicating whether AVX is supported.</returns>
    public static abstract bool IsAvxSupported();

    /// <summary>
    /// Invokes the vectorized operation on a Vector256 input.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<float> ApplyAvxFp32(Vector256<float> input);

    /// <summary>
    /// Invokes the vectorized operation on a Vector256 input.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<double> ApplyAvxFp64(Vector256<double> input);
}