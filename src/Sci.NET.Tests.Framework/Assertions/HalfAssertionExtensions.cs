// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Tests.Framework.Assertions;

/// <summary>
/// Assertion extensions for <see cref="Half" />.
/// </summary>
[PublicAPI]
public static class HalfAssertionExtensions
{
    /// <summary>
    /// Extension method to create a <see cref="Half" /> object for the given <see cref="Half" />.
    /// </summary>
    /// <param name="half">The <see cref="Half"/> to create assertions for.</param>
    /// <returns>A <see cref="HalfAssertions" /> object for the given <see cref="Half" />.</returns>
    public static HalfAssertions Should(this Half half)
    {
        return new(half);
    }
}