// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Memory;

namespace Sci.NET.Mathematics.UnitTests.Memory.SystemMemoryBlock;

public class UnsafeFreeMemoryShould
{
    [Fact]
    public void FreeMemory()
    {
        // Arrange
        using var block = new SystemMemoryBlock<int>(128);

        // Act
        block.UnsafeFreeMemory();

        // Assert
        block.IsDisposed.Should().BeTrue();
    }
}