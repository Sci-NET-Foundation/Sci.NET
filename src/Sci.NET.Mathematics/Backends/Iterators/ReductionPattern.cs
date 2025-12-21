// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Iterators;

/// <summary>
/// Defines the pattern of reduction to be performed on a tensor.
/// </summary>
public enum ReductionPattern
{
    /// <summary>
    /// Reducing all elements to a single scalar.
    /// </summary>
    FullReduction = 0,

    /// <summary>
    /// Reducing trailing (innermost) dimensions.
    /// </summary>
    ContiguousInner = 1,

    /// <summary>
    /// Reducing leading (outermost) dimensions.
    /// </summary>
    ContiguousOuter = 2,

    /// <summary>
    /// Reducing strided dimensions.
    /// </summary>
    Strided = 3
}