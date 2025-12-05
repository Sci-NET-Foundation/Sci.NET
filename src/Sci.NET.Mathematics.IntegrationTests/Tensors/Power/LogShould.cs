// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Power;

public class LogShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResultWithGrad_GivenVector(IDevice device)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[] { -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, requiresGradient: true);
        using var expectedResult = Tensor.FromArray<float>(
            new float[]
            {
                float.NaN, float.NegativeInfinity, 0.0f, 0.6931f, 1.0986f,
                1.3863f, 1.6094f, 1.7918f, 1.9459f, 2.0794f, 2.1972f, 2.3026f,
                2.3979f, 2.4849f, 2.5649f, 2.6391f, 2.7081f, 2.7726f
            });

        using var expectedGradient = Tensor.FromArray<float>(
            new float[]
            {
                -1.0f, float.PositiveInfinity,  1.0f,  0.5f,  0.3333f,  0.25f,  0.2f,  0.1667f,
                0.1429f,  0.1250f,  0.1111f,  0.1f,  0.0909f,  0.0833f,  0.0769f,  0.0714f,
                0.0667f,  0.0625f
            });

        tensor.To(device);
        expectedGradient.To(device);

        // Act
        using var result = tensor.Log();
        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), 1e-4f);
        tensor.Gradient!.Should().NotBeNull();
        tensor.Gradient?.Should().HaveApproximatelyEquivalentElements(expectedGradient.ToArray(), 1e-4f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalar(IDevice device)
    {
        // Log(2) = 0.693147180559945
        LogScalarTest<float>(2, device).Should().BeApproximately(0.693147180559945f, 1e-6f);
        LogScalarTest<double>(2, device).Should().BeApproximately(0.693147180559945d, 1e-6d);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVector(IDevice device)
    {
        LogVectorTest<float>(new float[] { 1, 2, 3 }, device)
            .Should()
            .BeEquivalentTo(new float[] { 0, 0.693147180559945f, 1.09861228866811f }, options => options.WithStrictOrdering());
        LogVectorTest<double>(new double[] { 1, 2, 3 }, device)
            .Should()
            .BeEquivalentTo(new double[] { 0, 0.6931471805599453, 1.0986122886681098 }, options => options.WithStrictOrdering());
    }

    private static Array LogVectorTest<TNumber>(TNumber[] values, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values).ToVector();

        tensor.To(device);

        return tensor.Log().ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrix(IDevice device)
    {
        LogMatrixTest<float>(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device)
            .Should()
            .BeEquivalentTo(new float[,] { { 0, 0.693147180559945f, 1.09861228866811f }, { 1.38629436111989f, 1.6094379124341f, 1.79175946922805f } }, options => options.WithStrictOrdering());
        LogMatrixTest<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device)
            .Should()
            .BeEquivalentTo(new double[,] { { 0, 0.6931471805599453, 1.0986122886681098 }, { 1.3862943611198906, 1.6094379124341003, 1.791759469228055 } }, options => options.WithStrictOrdering());
    }

    private static Array LogMatrixTest<TNumber>(TNumber[,] values, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values).ToMatrix();

        tensor.To(device);

        return tensor.Log().ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensor(IDevice device)
    {
        LogTensorTest<float>(new float[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device)
            .Should()
            .BeEquivalentTo(
                new float[,,]
                {
                    { { 0, 0.693147180559945f, 1.09861228866811f }, { 1.38629436111989f, 1.6094379124341f, 1.79175946922805f } },
                    { { 1.94591014905531f, 2.07944154167984f, 2.19722457733622f }, { 2.30258509299405f, 2.39789527279837f, 2.484906649788f } }
                },
                options => options.WithStrictOrdering());

        LogTensorTest<double>(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device)
            .Should()
            .BeEquivalentTo(
                new double[,,]
                {
                    { { 0, 0.6931471805599453, 1.0986122886681098 }, { 1.3862943611198906, 1.6094379124341003, 1.791759469228055 } },
                    { { 1.9459101490553132, 2.0794415416798357, 2.1972245773362196 }, { 2.302585092994046, 2.3978952727983707, 2.4849066497880004 } }
                },
                options => options.WithStrictOrdering());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResultAndGradient_GivenMatrix(IDevice device)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, requiresGradient: true);

        tensor.To(device);

        // Act
        using var result = tensor.Log();

        result.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(new float[,] { { 0.0000f, 0.6931f }, { 1.0986f, 1.3863f }, { 1.6094f, 1.7918f } }, 1e-4f);
        tensor.Gradient!.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(new float[,] { { 1F, 0.5F }, { 0.33333334F, 0.25F }, { 0.2F, 0.16666667F } });
    }

    private static TNumber LogScalarTest<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var scalar = new Scalar<TNumber>(value);

        scalar.To(device);

        return scalar.Log().Value;
    }

    private static Array LogTensorTest<TNumber>(TNumber[,,] values, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values);

        tensor.To(device);

        return tensor.Log().ToArray();
    }
}