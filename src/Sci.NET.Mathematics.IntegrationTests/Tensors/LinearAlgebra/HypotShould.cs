// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class HypotShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallBf16(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<BFloat16>(new BFloat16[] { 3.0f, 5.0f, 8.0f, 7.0f, 20.0f, 12.0f });
        var right = Tensor.FromArray<BFloat16>(new BFloat16[] { 4.0f, 12.0f, 15.0f, 24.0f, 21.0f, 35.0f });
        var expected = Tensor.FromArray<BFloat16>(new BFloat16[] { 5.0f, 13.0f, 17.0f, 25.0f, 29.0f, 37.0f });

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallFp16(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<Half>(new Half[] { (Half)3.0f, (Half)5.0f, (Half)8.0f, (Half)7.0f, (Half)20.0f, (Half)12.0f });
        var right = Tensor.FromArray<Half>(new Half[] { (Half)4.0f, (Half)12.0f, (Half)15.0f, (Half)24.0f, (Half)21.0f, (Half)35.0f });
        var expected = Tensor.FromArray<Half>(new Half[] { (Half)5.0f, (Half)13.0f, (Half)17.0f, (Half)25.0f, (Half)29.0f, (Half)37.0f });

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallFp32(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<float>(new float[] { 3.0f, 5.0f, 8.0f, 7.0f, 20.0f, 12.0f });
        var right = Tensor.FromArray<float>(new float[] { 4.0f, 12.0f, 15.0f, 24.0f, 21.0f, 35.0f });
        var expected = Tensor.FromArray<float>(new float[] { 5.0f, 13.0f, 17.0f, 25.0f, 29.0f, 37.0f });

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallFp64(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<double>(new double[] { 3.0f, 5.0f, 8.0f, 7.0f, 20.0f, 12.0f });
        var right = Tensor.FromArray<double>(new double[] { 4.0f, 12.0f, 15.0f, 24.0f, 21.0f, 35.0f });
        var expected = Tensor.FromArray<double>(new double[] { 5.0f, 13.0f, 17.0f, 25.0f, 29.0f, 37.0f });

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeFp16(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<Half>(Enumerable.Range(0, 1024).Select(x => (Half)x).ToArray());
        var right = Tensor.FromArray<Half>(Enumerable.Range(0, 1024).Select(x => (Half)(x * 2)).ToArray());
        var expected = Tensor.FromArray<Half>(Enumerable.Range(0, 1024).Select(x => (Half)Math.Sqrt((x * x) + (2 * x * 2 * x))).ToArray());

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeFp32(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<float>(Enumerable.Range(0, 1024).Select(x => (float)x).ToArray());
        var right = Tensor.FromArray<float>(Enumerable.Range(0, 1024).Select(x => (float)(x * 2)).ToArray());
        var expected = Tensor.FromArray<float>(Enumerable.Range(0, 1024).Select(x => (float)Math.Sqrt((x * x) + (2 * x * 2 * x))).ToArray());

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeFp64(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<double>(Enumerable.Range(0, 1024).Select(x => (double)x).ToArray());
        var right = Tensor.FromArray<double>(Enumerable.Range(0, 1024).Select(x => (double)(x * 2)).ToArray());
        var expected = Tensor.FromArray<double>(Enumerable.Range(0, 1024).Select(x => Math.Sqrt((x * x) + (2 * x * 2 * x))).ToArray());

        left.To(device);
        right.To(device);

        // Act
        var result = left.Hypot(right);

        // Assert
        result.Should().HaveEquivalentElements(expected.ToArray());
    }
}