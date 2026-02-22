// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// An interface for AVX/FMA binary operations. We assume that if FMA is supported, AVX is also supported.
/// In the rare case where AVX is supported but FMA is not, the scalar implementations will be used.
/// </summary>
public interface IBinaryOperationAvx2 : IBinaryOperationTail
{
    /// <summary>
    /// Determines whether AVX/FMA is supported on the current machine.
    /// </summary>
    /// <returns>A boolean indicating whether AVX is supported.</returns>
    public static abstract bool HasAvx2Implementation();

    /// <summary>
    /// Invokes the vectorized operation on two Vector256 inputs.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<float> ApplyAvxFp32(Vector256<float> left, Vector256<float> right);

    /// <summary>
    /// Invokes the vectorized operation on two Vector256 inputs.
    /// </summary>
    /// <param name="left">The left input.</param>
    /// <param name="right">The right input.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<double> ApplyAvxFp64(Vector256<double> left, Vector256<double> right);
}