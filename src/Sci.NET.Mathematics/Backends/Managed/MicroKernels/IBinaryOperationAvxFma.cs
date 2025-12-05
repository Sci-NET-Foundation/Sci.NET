// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Interface for AVX FMA binary operations.
/// </summary>
public interface IBinaryOperationAvxFma : IBinaryOperationAvxTail
{
    /// <summary>
    /// Determines whether AVX2 with FMA is supported on the current hardware.
    /// </summary>
    /// <returns>A value indicating whether AVX2 with FMA is supported.</returns>
    public static abstract bool IsAvxFmaSupported();

    /// <summary>
    /// Invokes the vectorized operation on two Vector256 inputs.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<float> ApplyAvxFmaFp32(Vector256<float> left, Vector256<float> right);

    /// <summary>
    /// Invokes the vectorized operation on two Vector256 inputs.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<double> ApplyAvxFmaFp64(Vector256<double> left, Vector256<double> right);
}