// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Random;

namespace Sci.NET.Mathematics.UnitTests.Random.PrngTests;

public class XavierNormalShould : PrngTestBase
{
    [Theory]
    [InlineData(5, 10)]
    [InlineData(10, 20)]
    [InlineData(25, 10)]
    public void NotBeEqual_WhenRunTwice_BFloat16(int fanIn, int fanOut)
    {
        foreach (var seed in Seeds)
        {
            // Arrange
            using var aMemory = new SystemMemoryBlock<BFloat16>(fanIn * fanOut);
            using var bMemory = new SystemMemoryBlock<BFloat16>(fanIn * fanOut);
            var prng = new Prng(seed);

            // Act
            prng.FillXavierNormal(aMemory, fanIn, fanOut);
            prng.FillXavierNormal(bMemory, fanIn, fanOut);

            // Assert
            var aArray = aMemory.ToArray();
            var bArray = bMemory.ToArray();

            aArray.Should().NotBeEquivalentTo(bArray);
        }
    }

    [Theory]
    [InlineData(5, 10)]
    [InlineData(10, 20)]
    [InlineData(25, 10)]
    public void NotBeEqual_WhenRunTwice_Float16(int fanIn, int fanOut)
    {
        foreach (var seed in Seeds)
        {
            // Arrange
            using var aMemory = new SystemMemoryBlock<Half>(fanIn * fanOut);
            using var bMemory = new SystemMemoryBlock<Half>(fanIn * fanOut);
            var prng = new Prng(seed);

            // Act
            prng.FillXavierNormal(aMemory, fanIn, fanOut);
            prng.FillXavierNormal(bMemory, fanIn, fanOut);

            // Assert
            var aArray = aMemory.ToArray();
            var bArray = bMemory.ToArray();

            aArray.Should().NotBeEquivalentTo(bArray);
        }
    }

    [Theory]
    [InlineData(5, 10)]
    [InlineData(10, 20)]
    [InlineData(25, 10)]
    public void NotBeEqual_WhenRunTwice_Float32(int fanIn, int fanOut)
    {
        foreach (var seed in Seeds)
        {
            // Arrange
            using var aMemory = new SystemMemoryBlock<float>(fanIn * fanOut);
            using var bMemory = new SystemMemoryBlock<float>(fanIn * fanOut);
            var prng = new Prng(seed);

            // Act
            prng.FillXavierNormal(aMemory, fanIn, fanOut);
            prng.FillXavierNormal(bMemory, fanIn, fanOut);

            // Assert
            var aArray = aMemory.ToArray();
            var bArray = bMemory.ToArray();

            aArray.Should().NotBeEquivalentTo(bArray);
        }
    }

    [Theory]
    [InlineData(5, 10)]
    [InlineData(10, 20)]
    [InlineData(25, 10)]
    public void NotBeEqual_WhenRunTwice_Float64(int fanIn, int fanOut)
    {
        foreach (var seed in Seeds)
        {
            // Arrange
            using var aMemory = new SystemMemoryBlock<double>(fanIn * fanOut);
            using var bMemory = new SystemMemoryBlock<double>(fanIn * fanOut);
            var prng = new Prng(seed);

            // Act
            prng.FillXavierNormal(aMemory, fanIn, fanOut);
            prng.FillXavierNormal(bMemory, fanIn, fanOut);

            // Assert
            var aArray = aMemory.ToArray();
            var bArray = bMemory.ToArray();

            aArray.Should().NotBeEquivalentTo(bArray);
        }
    }
}