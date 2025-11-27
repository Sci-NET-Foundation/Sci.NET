// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using FluentAssertions.Execution;
using FluentAssertions.Numeric;

namespace Sci.NET.Tests.Framework.Assertions;

/// <summary>
/// Assertions for <see cref="Half" />.
/// </summary>
public class HalfAssertions : NumericAssertions<Half, HalfAssertions>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="HalfAssertions"/> class.
    /// </summary>
    /// <param name="value">The value to create assertions for.</param>
    public HalfAssertions(Half value)
        : base(value)
    {
    }

    /// <summary>
    /// Asserts that the value is approximately equal to the expected value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="because">The reason why the assertion is needed. If the phrase does not start with the word <i>because</i>, it is prepended automatically.</param>
    /// <param name="becauseArgs">Zero or more objects to format using the placeholders in <paramref name="because" />.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    [PublicAPI]
    public new AndConstraint<HalfAssertions> Be(Half expected, string because = "", params object[] becauseArgs)
    {
        _ = Execute
            .Assertion
            .ForCondition(Subject?.Equals(expected) ?? false)
            .BecauseOf(because, becauseArgs)
            .FailWith("Expected {context:value} to be approximately {0}{reason}, but found {1}.", expected, Subject);

        return new AndConstraint<HalfAssertions>(this);
    }

    /// <summary>
    /// Asserts that the value is approximately equal to the expected value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="tolerance">The tolerance within which the value is expected to be.</param>
    /// <param name="because">The reason why the assertion is needed. If the phrase does not start with the word <i>because</i>, it is prepended automatically.</param>
    /// <param name="becauseArgs">Zero or more objects to format using the placeholders in <paramref name="because" />.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the subject is <see langword="null" /> or not comparable (I.E. infinite, NaN).</exception>
    [PublicAPI]
    public AndConstraint<HalfAssertions> BeApproximately(Half expected, float tolerance, string because = "", params object[] becauseArgs)
    {
        if (!Subject.HasValue)
        {
            throw new InvalidOperationException("Cannot assert that a null value is approximately equal to another value.");
        }

        if (Half.IsNaN(Subject.Value))
        {
            throw new InvalidOperationException("Cannot assert that a NaN value is approximately equal to another value.");
        }

        if (Half.IsNaN(expected))
        {
            throw new InvalidOperationException("Cannot assert that a value is approximately equal to a NaN value.");
        }

        if (Half.IsPositiveInfinity(Subject.Value) || Half.IsNegativeInfinity(Subject.Value))
        {
            throw new InvalidOperationException("Cannot assert that an infinity value is approximately equal to another value.");
        }

        if (Half.IsPositiveInfinity(expected) || Half.IsNegativeInfinity(expected))
        {
            throw new InvalidOperationException("Cannot assert that a value is approximately equal to an infinity value.");
        }

        _ = Execute
            .Assertion
            .ForCondition(float.Abs((float)Subject.Value - (float)expected) <= tolerance)
            .BecauseOf(because, becauseArgs)
            .FailWith(
                "Expected {context:value} to be approximately {0} +/- {1}{reason}, but found {2}.",
                expected,
                tolerance,
                Subject);

        return new AndConstraint<HalfAssertions>(this);
    }
}