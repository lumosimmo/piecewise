use alloy_primitives::{U256, U512, uint};
use serde::{Deserialize, Serialize};

pub const ONE_E6: U256 = uint!(1000000_U256);
pub const ONE_E15: U256 = uint!(1000000000000000_U256);
pub const ONE_E18: U256 = uint!(1000000000000000000_U256);
pub const ONE_E27: U256 = uint!(1000000000000000000000000000_U256);
pub const ONE_E9: U256 = uint!(1000000000_U256);
pub const ONE_E36: U256 = uint!(1000000000000000000000000000000000000_U256);
pub const ONE_E54: U256 = uint!(1000000000000000000000000000000000000000000000000000000_U256);
/// 2^248 - 1
pub const U248_MAX: U256 = U256::from_limbs([u64::MAX, u64::MAX, u64::MAX, 0x00ff_ffff_ffff_ffff]);

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum MathError {
    DivisionByZero,
    Overflow,
}

/// Returns the division of two numbers, rounding towards infinity.
pub fn ceil_div(a: U256, b: U256) -> Result<U256, MathError> {
    if b.is_zero() {
        return Err(MathError::DivisionByZero);
    }
    if a.is_zero() {
        return Ok(U256::ZERO);
    }
    // Operate on the U512 space to simplify the formula
    let a512 = U512::from(a);
    let b512 = U512::from(b);
    Ok(((a512 + b512 - U512::ONE) / b512).to::<U256>())
}

/// Calculates (x * y / denominator) with full precision, rounding towards zero.
pub fn mul_div_floor(x: U256, y: U256, denominator: U256) -> Result<U256, MathError> {
    if denominator.is_zero() {
        return Err(MathError::DivisionByZero);
    }
    let prod = U512::from(x) * U512::from(y);
    let result = prod / U512::from(denominator);
    if result > U512::from(U256::MAX) {
        return Err(MathError::Overflow);
    }
    Ok(result.to::<U256>())
}

/// Calculates (x * y / denominator) with full precision, rounding towards infinity.
pub fn mul_div_ceil(x: U256, y: U256, denominator: U256) -> Result<U256, MathError> {
    if denominator.is_zero() {
        return Err(MathError::DivisionByZero);
    }
    let prod = U512::from(x) * U512::from(y);
    let denominator_512 = U512::from(denominator);
    let mut result = prod / denominator_512;
    if prod % denominator_512 != U512::ZERO {
        result += U512::ONE;
    }
    if result > U512::from(U256::MAX) {
        return Err(MathError::Overflow);
    }
    Ok(result.to::<U256>())
}

/// Returns the square root of a number, rounding towards zero.
///
/// This method is based on Newton's method for computing square roots; the algorithm is restricted
/// to only using integer operations.
pub fn sqrt_floor(a: U256) -> Result<U256, MathError> {
    if a <= U256::from(1) {
        return Ok(a);
    }

    let mut xn = U256::from(1) << (a.log2() / 2);
    xn = (U256::from(3) * xn) >> 1;

    // Newton's method iterations.
    xn = (xn + a / xn) >> 1;
    xn = (xn + a / xn) >> 1;
    xn = (xn + a / xn) >> 1;
    xn = (xn + a / xn) >> 1;
    xn = (xn + a / xn) >> 1;
    xn = (xn + a / xn) >> 1;

    if xn > a / xn {
        Ok(xn - U256::ONE)
    } else {
        Ok(xn)
    }
}

/// Returns the square root of a number, rounding towards infinity.
pub fn sqrt_ceil(a: U256) -> Result<U256, MathError> {
    let result = sqrt_floor(a)?;
    if result * result < a {
        Ok(result + U256::ONE)
    } else {
        Ok(result)
    }
}

/// Calculates the ratio of the absolute difference between two values to the second value, scaled
/// to a given precision.
pub fn delta_ratio(left: U256, right: U256, precision: U256) -> Result<U256, MathError> {
    let diff = if left > right {
        left - right
    } else {
        right - left
    };
    mul_div_floor(diff, precision, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use std::str::FromStr;

    #[test]
    fn test_ceil_div_by_zero() {
        assert_eq!(
            ceil_div(U256::from(2), U256::ZERO).unwrap_err(),
            MathError::DivisionByZero
        );
    }

    #[test]
    fn test_ceil_div() {
        assert_eq!(ceil_div(U256::ZERO, U256::from(2)).unwrap(), U256::ZERO);
        assert_eq!(
            ceil_div(U256::from(10), U256::from(5)).unwrap(),
            U256::from(2)
        );
        assert_eq!(
            ceil_div(U256::from(42), U256::from(13)).unwrap(),
            U256::from(4)
        );
        assert_eq!(
            ceil_div(U256::MAX, U256::from(2)).unwrap(),
            U256::from(1) << 255
        );
        assert_eq!(ceil_div(U256::MAX, U256::from(1)).unwrap(), U256::MAX);
    }

    #[test]
    fn test_mul_div_floor_by_zero() {
        assert_eq!(
            mul_div_floor(U256::from(1), U256::from(1), U256::ZERO).unwrap_err(),
            MathError::DivisionByZero
        );
    }

    #[test]
    fn test_mul_div_floor_overflow() {
        assert_eq!(
            mul_div_floor(U256::from(5), U256::MAX, U256::from(2)).unwrap_err(),
            MathError::Overflow
        );
    }

    #[test]
    fn test_mul_div_floor() {
        assert_eq!(
            mul_div_floor(U256::from(3), U256::from(4), U256::from(5)).unwrap(),
            U256::from(2)
        );
        assert_eq!(
            mul_div_floor(U256::from(3), U256::from(5), U256::from(5)).unwrap(),
            U256::from(3)
        );
        assert_eq!(
            mul_div_floor(U256::from(42), U256::MAX - U256::from(1), U256::MAX).unwrap(),
            U256::from(41)
        );
        assert_eq!(
            mul_div_floor(U256::from(17), U256::MAX, U256::MAX).unwrap(),
            U256::from(17)
        );
        assert_eq!(
            mul_div_floor(
                U256::MAX - U256::from(1),
                U256::MAX - U256::from(1),
                U256::MAX
            )
            .unwrap(),
            U256::MAX - U256::from(2)
        );
        assert_eq!(
            mul_div_floor(U256::MAX, U256::MAX - U256::from(1), U256::MAX).unwrap(),
            U256::MAX - U256::from(1)
        );
        assert_eq!(
            mul_div_floor(U256::MAX, U256::MAX, U256::MAX).unwrap(),
            U256::MAX
        );
    }

    #[test]
    fn test_mul_div_ceil_by_zero() {
        assert_eq!(
            mul_div_ceil(U256::from(1), U256::from(1), U256::ZERO).unwrap_err(),
            MathError::DivisionByZero
        );
    }

    #[test]
    fn test_mul_div_ceil_overflow() {
        assert_eq!(
            mul_div_ceil(U256::from(5), U256::MAX, U256::from(2)).unwrap_err(),
            MathError::Overflow
        );
    }

    #[test]
    fn test_mul_div_ceil() {
        assert_eq!(
            mul_div_ceil(U256::from(3), U256::from(4), U256::from(5)).unwrap(),
            U256::from(3)
        );
        assert_eq!(
            mul_div_ceil(U256::from(3), U256::from(5), U256::from(5)).unwrap(),
            U256::from(3)
        );
        assert_eq!(
            mul_div_ceil(U256::from(42), U256::MAX - U256::from(1), U256::MAX).unwrap(),
            U256::from(42)
        );
        assert_eq!(
            mul_div_ceil(U256::from(17), U256::MAX, U256::MAX).unwrap(),
            U256::from(17)
        );
        assert_eq!(
            mul_div_ceil(
                U256::MAX - U256::from(1),
                U256::MAX - U256::from(1),
                U256::MAX
            )
            .unwrap(),
            U256::MAX - U256::from(1)
        );
        assert_eq!(
            mul_div_ceil(U256::MAX, U256::MAX - U256::from(1), U256::MAX).unwrap(),
            U256::MAX - U256::from(1)
        );
        assert_eq!(
            mul_div_ceil(U256::MAX, U256::MAX, U256::MAX).unwrap(),
            U256::MAX
        );
    }

    #[test]
    fn test_sqrt_floor() {
        assert_eq!(sqrt_floor(U256::ZERO).unwrap(), U256::ZERO);
        assert_eq!(sqrt_floor(U256::from(1)).unwrap(), U256::from(1));
        assert_eq!(sqrt_floor(U256::from(2)).unwrap(), U256::from(1));
        assert_eq!(sqrt_floor(U256::from(3)).unwrap(), U256::from(1));
        assert_eq!(sqrt_floor(U256::from(4)).unwrap(), U256::from(2));
        assert_eq!(sqrt_floor(U256::from(144)).unwrap(), U256::from(12));
        assert_eq!(sqrt_floor(U256::from(999999)).unwrap(), U256::from(999));
        assert_eq!(sqrt_floor(U256::from(1000000)).unwrap(), U256::from(1000));
        assert_eq!(sqrt_floor(U256::from(1000001)).unwrap(), U256::from(1000));
        assert_eq!(sqrt_floor(U256::from(1002000)).unwrap(), U256::from(1000));
        assert_eq!(sqrt_floor(U256::from(1002001)).unwrap(), U256::from(1001));
        assert_eq!(
            sqrt_floor(U256::MAX).unwrap(),
            U256::from_str("340282366920938463463374607431768211455").unwrap()
        );
    }

    #[test]
    fn test_sqrt_ceil() {
        assert_eq!(sqrt_ceil(U256::ZERO).unwrap(), U256::ZERO);
        assert_eq!(sqrt_ceil(U256::from(1)).unwrap(), U256::from(1));
        assert_eq!(sqrt_ceil(U256::from(2)).unwrap(), U256::from(2));
        assert_eq!(sqrt_ceil(U256::from(3)).unwrap(), U256::from(2));
        assert_eq!(sqrt_ceil(U256::from(4)).unwrap(), U256::from(2));
        assert_eq!(sqrt_ceil(U256::from(144)).unwrap(), U256::from(12));
        assert_eq!(sqrt_ceil(U256::from(999999)).unwrap(), U256::from(1000));
        assert_eq!(sqrt_ceil(U256::from(1000000)).unwrap(), U256::from(1000));
        assert_eq!(sqrt_ceil(U256::from(1000001)).unwrap(), U256::from(1001));
        assert_eq!(sqrt_ceil(U256::from(1002000)).unwrap(), U256::from(1001));
        assert_eq!(sqrt_ceil(U256::from(1002001)).unwrap(), U256::from(1001));
        assert_eq!(
            sqrt_ceil(U256::MAX).unwrap(),
            U256::from_str("340282366920938463463374607431768211456").unwrap()
        );
    }

    #[test]
    fn test_delta_ratio() {
        let precision = ONE_E18;
        assert_eq!(
            delta_ratio(U256::from(120), U256::from(100), precision).unwrap(),
            precision / U256::from(5) // 20%
        );
        assert_eq!(
            delta_ratio(U256::from(80), U256::from(100), precision).unwrap(),
            precision / U256::from(5) // 20%
        );
        assert_eq!(
            delta_ratio(U256::from(100), U256::from(100), precision).unwrap(),
            U256::ZERO
        );
    }
}
