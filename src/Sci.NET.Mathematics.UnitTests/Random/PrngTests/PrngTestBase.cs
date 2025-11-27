// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.UnitTests.Random.PrngTests;

public abstract class PrngTestBase
{
    public static readonly TheoryData<ulong> Seeds =
    [
        0,
        1,
        65456684,
        455577877,
        18446744073709551615
    ];
}