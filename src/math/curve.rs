use crate::math::common::{
    MathError, ONE_E9, ONE_E15, ONE_E18, ONE_E27, ONE_E36, ONE_E54, U248_MAX, ceil_div,
    delta_ratio, mul_div_ceil, sqrt_ceil,
};
use crate::math::quote::find_reserve1_for_reserve0;
use alloy_primitives::{Address, I256, U256, aliases::U112};
use serde::{Deserialize, Serialize};

/// Parameters for the EulerSwap pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EulerSwapParams {
    pub vault0: Address,
    pub vault1: Address,
    pub euler_account: Address,
    pub equilibrium_reserve0: U112,
    pub equilibrium_reserve1: U112,
    pub price_x: U256,
    pub price_y: U256,
    pub concentration_x: U256,
    pub concentration_y: U256,
    pub fee: U256,
    pub protocol_fee: U256,
    pub protocol_fee_recipient: Address,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum CurveError {
    Math(MathError),
    PriceBelowApex,
    NoSolution,
    SwapLimitExceeded,
}

impl From<MathError> for CurveError {
    fn from(e: MathError) -> Self {
        CurveError::Math(e)
    }
}

/// Computes the output `y` for a given input `x` for the EulerSwap curve.
///
/// We are translating the logic faithfully, assuming inputs are within constraints that prevent
/// overflow, similar to the original implementation. The mul_div_ceil function uses U512 internally
/// to prevent overflow for the multiplication part. Such are the other functions in this file.
pub fn f(x: U256, px: U256, py: U256, x0: U256, y0: U256, c: U256) -> Result<U256, CurveError> {
    let v = mul_div_ceil(px * (x0 - x), c * x + (ONE_E18 - c) * x0, x * ONE_E18)?;
    if v > U248_MAX {
        return Err(CurveError::Math(MathError::Overflow));
    }
    Ok(y0 + ceil_div(v, py)?)
}

/// Computes the output `x` for a given input `y` for the EulerSwap inverse curve.
///
/// We are translating the logic faithfully, assuming inputs are within constraints that prevent
/// overflow, similar to the original implementation. The mul_div_ceil function uses U512 internally
/// to prevent overflow for the multiplication part. Such are the other functions in this file.
pub fn f_inverse(
    y: U256,
    px: U256,
    py: U256,
    x0: U256,
    y0: U256,
    c: U256,
) -> Result<U256, CurveError> {
    let term1_unsigned = mul_div_ceil(py * ONE_E18, y - y0, px)?;
    let term1 = I256::from_raw(term1_unsigned); // scale: 1e36

    let i_c = I256::from_raw(c);
    let i_one_e18 = I256::from_raw(ONE_E18);
    let i_x0 = I256::from_raw(x0);
    let term2 = (I256::from(U256::from(2u64)) * i_c - i_one_e18) * i_x0; // scale: 1e36

    let b: I256 = (term1 - term2) / i_one_e18; // scale: 1e18
    let c_quad: U256 = mul_div_ceil(ONE_E18 - c, x0 * x0, ONE_E18)?; // scale: 1e36
    let four_ac: U256 = mul_div_ceil(U256::from(4) * c, c_quad, ONE_E18)?; // scale: 1e36

    let abs_b = b.abs().into_raw();
    let squared_b: U256;
    let discriminant: U256;
    let mut sqrt: U256;

    if abs_b < ONE_E36 {
        squared_b = abs_b * abs_b; // scale: 1e36
        discriminant = squared_b + four_ac; // scale: 1e36
        sqrt = sqrt_ceil(discriminant)?; // scale: 1e18
    } else {
        // If B^2 cannot be calculated directly at 1e18 scale without overflowing, we need to scale
        // down the input to prevent overflow.
        let scale = compute_scale(abs_b)?;
        squared_b = mul_div_ceil(abs_b / scale, abs_b, scale)?;
        discriminant = squared_b + four_ac / (scale * scale);
        sqrt = sqrt_ceil(discriminant)?;
        sqrt *= scale;
    }

    let x: U256 = if b.is_negative() || b.is_zero() {
        mul_div_ceil(abs_b + sqrt, ONE_E18, U256::from(2) * c)? + U256::ONE
    } else {
        ceil_div(U256::from(2) * c_quad, abs_b + sqrt)? + U256::ONE
    };

    if x >= x0 { Ok(x0) } else { Ok(x) }
}

/// First derivative -df/dx on a given branch, taking the positive branch, in WEI precision.
///
/// Numerically equal to the marginal spot price (in 1e18 fixed-point, aka WEI).
pub fn df_dx(x: U256, px: U256, py: U256, x0: U256, cx: U256) -> Result<U256, CurveError> {
    let r0 = mul_div_ceil(x0, x0, x)?;
    let r = mul_div_ceil(r0, ONE_E18, x)?;
    let term = mul_div_ceil(ONE_E18 - cx, r, ONE_E18)?;
    let inner_expr = cx + term;

    Ok(mul_div_ceil(px, inner_expr, py)?)
}

/// First derivative -df/dx on a given branch, taking the positive branch, in RAY precision.
///
/// Numerically equal to the marginal spot price (in 1e27 fixed-point, aka RAY).
pub fn df_dx_ray(x: U256, px: U256, py: U256, x0: U256, cx: U256) -> Result<U256, CurveError> {
    let r0 = mul_div_ceil(x0, x0, x)?;
    let r = mul_div_ceil(r0, ONE_E18, x)?;
    let term = mul_div_ceil(ONE_E18 - cx, r, ONE_E18)?;
    let inner_expr = cx + term;
    Ok(mul_div_ceil(px * ONE_E9, inner_expr, py)?)
}

/// Computes an optimal scaling factor for a large number to avoid overflow during squaring.
///
/// This is a utility function used by `f_inverse` when calculating the discriminant.
fn compute_scale(x: U256) -> Result<U256, CurveError> {
    let bits = if x.is_zero() { 0 } else { x.log2() + 1 };

    if bits > 128 {
        let excess_bits = bits - 128;
        Ok(U256::from(1) << excess_bits)
    } else {
        Ok(U256::ONE)
    }
}

/// Returns true if the specified reserve amounts would be acceptable, false otherwise.
/// Acceptable points are on, or above and to-the-right of the swapping curve.
pub fn verify(
    p: &EulerSwapParams,
    new_reserve0: U256,
    new_reserve1: U256,
) -> Result<bool, CurveError> {
    if new_reserve0 > U256::from(U112::MAX) || new_reserve1 > U256::from(U112::MAX) {
        return Ok(false);
    }

    let equilibrium_reserve0 = U256::from(p.equilibrium_reserve0);
    let equilibrium_reserve1 = U256::from(p.equilibrium_reserve1);

    if new_reserve0 >= equilibrium_reserve0 {
        if new_reserve1 >= equilibrium_reserve1 {
            return Ok(true);
        }
        Ok(new_reserve0
            >= f(
                new_reserve1,
                p.price_y,
                p.price_x,
                equilibrium_reserve1,
                equilibrium_reserve0,
                p.concentration_y,
            )?)
    } else {
        if new_reserve1 < equilibrium_reserve1 {
            return Ok(false);
        }
        Ok(new_reserve1
            >= f(
                new_reserve0,
                p.price_x,
                p.price_y,
                equilibrium_reserve0,
                equilibrium_reserve1,
                p.concentration_x,
            )?)
    }
}

/// Computes the marginal price at the given reserve vector, in WEI precision.
pub fn get_current_price(
    p: &EulerSwapParams,
    reserve0: U256,
    reserve1: U256,
) -> Result<U256, CurveError> {
    let equilibrium_reserve0 = U256::from(p.equilibrium_reserve0);
    let equilibrium_reserve1 = U256::from(p.equilibrium_reserve1);

    if reserve0 <= equilibrium_reserve0 {
        // We are on or to the left of the apex -> slope directly gives X‑price.
        if reserve0 == equilibrium_reserve0 {
            return Ok(mul_div_ceil(p.price_x, ONE_E18, p.price_y)?);
        }
        df_dx(
            reserve0,
            p.price_x,
            p.price_y,
            equilibrium_reserve0,
            p.concentration_x,
        )
    } else {
        // If on the right branch, derive the slope in Y‑space and invert.
        if reserve1 == equilibrium_reserve1 {
            return Ok(mul_div_ceil(p.price_y, ONE_E18, p.price_x)?);
        }
        let price = df_dx(
            reserve1,
            p.price_y,
            p.price_x,
            equilibrium_reserve1,
            p.concentration_y,
        )?;
        Ok(ceil_div(ONE_E36, price)?) // reciprocal because dx/dy = 1/(dy/dx)
    }
}

/// Computes the marginal price at the given reserve vector, in RAY precision.
pub fn get_current_price_ray(
    p: &EulerSwapParams,
    reserve0: U256,
    reserve1: U256,
) -> Result<U256, CurveError> {
    let equilibrium_reserve0 = U256::from(p.equilibrium_reserve0);
    let equilibrium_reserve1 = U256::from(p.equilibrium_reserve1);

    if reserve0 <= equilibrium_reserve0 {
        if reserve0 == equilibrium_reserve0 {
            return Ok(mul_div_ceil(p.price_x, ONE_E27, p.price_y)?);
        }
        df_dx_ray(
            reserve0,
            p.price_x,
            p.price_y,
            equilibrium_reserve0,
            p.concentration_x,
        )
    } else {
        if reserve1 == equilibrium_reserve1 {
            return Ok(mul_div_ceil(p.price_y, ONE_E27, p.price_x)?);
        }
        let price = df_dx_ray(
            reserve1,
            p.price_y,
            p.price_x,
            equilibrium_reserve1,
            p.concentration_y,
        )?;
        Ok(ceil_div(ONE_E54, price)?)
    }
}

/// Finds the unique lattice point `(reserve0, reserve1)` whose marginal price equals to a given
/// price `current_price`, in WEI precision. Binary search is used.
pub fn get_current_reserves(
    p: &EulerSwapParams,
    current_price: U256,
) -> Result<(U256, U256), CurveError> {
    let x0 = U256::from(p.equilibrium_reserve0);
    let y0 = U256::from(p.equilibrium_reserve1);

    let apex_price = mul_div_ceil(p.price_x, ONE_E18, p.price_y)?;
    if current_price == apex_price {
        return Ok((x0, y0));
    }

    if current_price > apex_price {
        let mut lo = if x0.is_zero() { U256::ZERO } else { U256::ONE };
        let mut hi = if x0.is_zero() {
            U256::ZERO
        } else {
            x0 - U256::ONE
        };

        while lo <= hi {
            let reserve0 = (lo + hi) >> 1;
            let reserve1 = find_reserve1_for_reserve0(p, reserve0)?;
            let price = get_current_price(p, reserve0, reserve1)?;

            let delta = delta_ratio(current_price, price, ONE_E15)?;
            if delta == U256::ZERO {
                return Ok((reserve0, reserve1));
            }
            if price > current_price {
                lo = reserve0 + U256::ONE;
            } else {
                if reserve0.is_zero() {
                    break;
                }
                hi = reserve0 - U256::ONE;
            }
        }
    }

    if current_price < apex_price {
        let mut lo = x0 + U256::ONE;
        let mut hi = U256::from(U112::MAX);

        while lo <= hi {
            let reserve0 = (lo + hi) >> 1;
            let reserve1 = find_reserve1_for_reserve0(p, reserve0)?;
            let price = get_current_price(p, reserve0, reserve1)?;

            let delta = delta_ratio(current_price, price, ONE_E15)?;
            if delta == U256::ZERO {
                return Ok((reserve0, reserve1));
            }
            if price < current_price {
                lo = reserve0 + U256::ONE;
            } else {
                if reserve0 == x0 + U256::ONE {
                    break;
                }
                hi = reserve0 - U256::ONE;
            }
        }
    }

    Err(CurveError::NoSolution)
}

/// Finds the unique lattice point `(reserve0, reserve1)` whose marginal price equals to a given
/// price `current_price_ray`, in RAY precision. Binary search is used.
pub fn get_current_reserves_ray(
    p: &EulerSwapParams,
    current_price_ray: U256,
) -> Result<(U256, U256), CurveError> {
    let x0 = U256::from(p.equilibrium_reserve0);
    let y0 = U256::from(p.equilibrium_reserve1);

    let apex_price = mul_div_ceil(p.price_x, ONE_E27, p.price_y)?;
    if current_price_ray == apex_price {
        return Ok((x0, y0));
    }

    if current_price_ray > apex_price {
        let mut lo = if x0.is_zero() { U256::ZERO } else { U256::ONE };
        let mut hi = if x0.is_zero() {
            U256::ZERO
        } else {
            x0 - U256::ONE
        };

        while lo <= hi {
            let reserve0 = (lo + hi) >> 1;
            let reserve1 = find_reserve1_for_reserve0(p, reserve0)?;
            let price = get_current_price_ray(p, reserve0, reserve1)?;

            // Sometimes we can't hit the exact price, so we have to exit when it's close enough.
            if delta_ratio(current_price_ray, price, ONE_E18)? == U256::ZERO {
                return Ok((reserve0, reserve1));
            }
            if price > current_price_ray {
                lo = reserve0 + U256::ONE;
            } else {
                if reserve0.is_zero() {
                    break;
                }
                hi = reserve0 - U256::ONE;
            }
        }
    }

    if current_price_ray < apex_price {
        let mut lo = x0 + U256::ONE;
        let mut hi = U256::from(U112::MAX);

        while lo <= hi {
            let reserve0 = (lo + hi) >> 1;
            let reserve1 = find_reserve1_for_reserve0(p, reserve0)?;
            let price = get_current_price_ray(p, reserve0, reserve1)?;

            if delta_ratio(current_price_ray, price, ONE_E18)? == U256::ZERO {
                return Ok((reserve0, reserve1));
            }
            if price < current_price_ray {
                lo = reserve0 + U256::ONE;
            } else {
                if reserve0 == x0 + U256::ONE {
                    break;
                }
                hi = reserve0 - U256::ONE;
            }
        }
    }

    Err(CurveError::NoSolution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{address, uint};
    use std::str::FromStr;

    #[test]
    fn test_f() {
        let x = uint!(100_U256) * ONE_E18;
        let px = uint!(1_U256) * ONE_E18;
        let py = uint!(1_U256) * ONE_E18;
        let x0 = uint!(200_U256) * ONE_E18;
        let y0 = uint!(50_U256) * ONE_E18;
        let c = uint!(1_U256) * ONE_E18;
        // Expected value is calculated based on the formula.
        // v = mulDiv(px * (x0 - x), c * x + (1e18 - c) * x0, x * 1e18, Ceil)
        // v = mulDiv(1e18 * 100e18, 1e18 * 100e18 + 0, 100e18 * 1e18, Ceil) = 100e18
        // y = y0 + (v + (py - 1)) / py = 50e18 + (100e18 + 1e18 - 1) / 1e18 = 50e18 + 100e18 = 150e18
        let expected_y = uint!(150_U256) * ONE_E18;
        assert_eq!(f(x, px, py, x0, y0, c).unwrap(), expected_y);
    }

    #[test]
    fn test_f_inverse() {
        // Using a test case from my PR to the official JS math lib.
        let y = uint!(350000000000000000000_U256);
        let px = uint!(1300000000000000000_U256);
        let py = uint!(1000000000000000000_U256);
        let x0 = uint!(170000000000000000000_U256);
        let y0 = uint!(100000000000000000000_U256);
        let c = uint!(100000000000000000_U256);
        // We expect to get back the original x value, or something very close to it.
        let expected_x = uint!(77399734182972528597_U256);
        let result_x = f_inverse(y, px, py, x0, y0, c).unwrap();
        // The inverse function might have precision errors, so we check if it's within a small delta.
        let delta = if result_x > expected_x {
            result_x - expected_x
        } else {
            expected_x - result_x
        };
        assert!(delta <= U256::from(1), "x value deviates too much");
    }

    #[test]
    fn test_f_and_f_inverse_roundtrip() {
        let x = uint!(12345_U256) * ONE_E18 / uint!(100_U256);
        let px = uint!(2_U256) * ONE_E18;
        let py = uint!(3_U256) * ONE_E18;
        let x0 = uint!(50000_U256) * ONE_E18;
        let y0 = uint!(10000_U256) * ONE_E18;
        let c = ONE_E18 / U256::from(2); // 0.5 * 1e18

        let y = f(x, px, py, x0, y0, c).unwrap();
        let recovered_x = f_inverse(y, px, py, x0, y0, c).unwrap();

        let delta = if recovered_x > x {
            recovered_x - x
        } else {
            x - recovered_x
        };
        assert!(delta <= U256::from(1), "x value deviates too much");
    }

    #[test]
    fn test_df_dx() {
        let x = uint!(100_U256) * ONE_E18;
        let px = uint!(1_U256) * ONE_E18;
        let py = uint!(1_U256) * ONE_E18;
        let x0 = uint!(200_U256) * ONE_E18;
        let cx = ONE_E18 / U256::from(10); // 0.1 * 1e18

        // r = (x0^2 / x^2) * 1e18 = ((200e18)^2 / (100e18)^2) * 1e18 = 4e18
        // inner = cx + (1e18 - cx) * r / 1e18
        //       = 0.1e18 + (0.9e18 * 4e18) / 1e18
        //       = 0.1e18 + 3.6e18 = 3.7e18
        // result = px * inner / py = 1e18 * 3.7e18 / 1e18 = 3.7e18
        let expected_df_dx = uint!(37_U256) * ONE_E18 / U256::from(10);
        assert_eq!(df_dx(x, px, py, x0, cx).unwrap(), expected_df_dx);
    }

    #[test]
    fn test_compute_scale() {
        // bits <= 128, scale should be 1
        assert_eq!(compute_scale(U256::from(1) << 127).unwrap(), U256::ONE);
        assert_eq!(
            compute_scale(U256::from_str("0xffffffffffffffffffffffffffffffff").unwrap()).unwrap(),
            U256::ONE
        );

        // bits > 128
        assert_eq!(compute_scale(U256::from(1) << 128).unwrap(), U256::from(2));
        assert_eq!(compute_scale(U256::from(1) << 129).unwrap(), U256::from(4));
        assert_eq!(compute_scale(U256::from(1) << 130).unwrap(), U256::from(8));
        assert_eq!(
            compute_scale(U256::MAX).unwrap(),
            U256::from(1) << (256 - 128)
        );
    }

    #[test]
    fn test_verify() {
        let eq_reserve_u256 = U256::from(100) * ONE_E18;
        let limbs = eq_reserve_u256.into_limbs();
        let eq_reserve_u112 = U112::from_limbs([limbs[0], limbs[1]]);
        let params = EulerSwapParams {
            vault0: Address::ZERO,
            vault1: Address::ZERO,
            euler_account: Address::ZERO,
            equilibrium_reserve0: eq_reserve_u112,
            equilibrium_reserve1: eq_reserve_u112,
            price_x: ONE_E18,
            price_y: ONE_E18,
            concentration_x: ONE_E18,
            concentration_y: ONE_E18,
            fee: U256::ZERO,
            protocol_fee: U256::ZERO,
            protocol_fee_recipient: Address::ZERO,
        };

        // Case 1: new reserves are above equilibrium
        assert!(
            verify(
                &params,
                eq_reserve_u256 + U256::from(1),
                eq_reserve_u256 + U256::from(1)
            )
            .unwrap()
        );

        // Case 2: new_reserve0 >= eq0, new_reserve1 < eq1.
        // This case checks new_reserve0 >= f(new_reserve1, ...)
        let reserve0_case2 = U256::from(110) * ONE_E18;
        let reserve1_case2 = U256::from(90) * ONE_E18; // 110+90 = 200, so it's on the curve.
        assert!(verify(&params, reserve0_case2, reserve1_case2).unwrap());
        // Point below the curve
        assert!(!verify(&params, reserve0_case2 - U256::from(1), reserve1_case2).unwrap());
        // Point above the curve
        assert!(verify(&params, reserve0_case2 + U256::from(1), reserve1_case2).unwrap());

        // Case 3: new_reserve0 < eq0.
        // Must have new_reserve1 >= eq1 to proceed, otherwise false.
        assert!(!verify(&params, U256::from(90) * ONE_E18, U256::from(90) * ONE_E18).unwrap());

        // This case checks new_reserve1 >= f(new_reserve0, ...)
        let reserve0_case3 = U256::from(90) * ONE_E18;
        let reserve1_case3 = U256::from(110) * ONE_E18; // 90+110 = 200, so it's on the curve.
        assert!(verify(&params, reserve0_case3, reserve1_case3).unwrap());
        // Point below the curve
        assert!(!verify(&params, reserve0_case3, reserve1_case3 - U256::from(1)).unwrap());
        // Point above the curve
        assert!(verify(&params, reserve0_case3, reserve1_case3 + U256::from(1)).unwrap());

        // Case 4: reserves exceed U112::MAX
        let large_reserve = U256::from(U112::MAX) + U256::from(1);
        assert!(!verify(&params, large_reserve, eq_reserve_u256).unwrap());
        assert!(!verify(&params, eq_reserve_u256, large_reserve).unwrap());
    }

    #[test]
    fn test_get_current_price() {
        let eq_reserve_u256 = U256::from(100) * ONE_E18;
        let limbs = eq_reserve_u256.into_limbs();
        let eq_reserve_u112 = U112::from_limbs([limbs[0], limbs[1]]);
        let params = EulerSwapParams {
            vault0: Address::ZERO,
            vault1: Address::ZERO,
            euler_account: Address::ZERO,
            equilibrium_reserve0: eq_reserve_u112,
            equilibrium_reserve1: eq_reserve_u112,
            price_x: U256::from(2) * ONE_E18,
            price_y: U256::from(1) * ONE_E18,
            concentration_x: ONE_E18 / U256::from(10), // 0.1
            concentration_y: ONE_E18 / U256::from(5),  // 0.2
            fee: U256::ZERO,
            protocol_fee: U256::ZERO,
            protocol_fee_recipient: Address::ZERO,
        };

        // Case 1: reserve0 < equilibrium_reserve0
        let reserve0_left = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_left = get_current_price(&params, reserve0_left, U256::ZERO).unwrap();
        let expected_price_left = df_dx(
            reserve0_left,
            params.price_x,
            params.price_y,
            eq_reserve_u256,
            params.concentration_x,
        )
        .unwrap();
        assert_eq!(price_left, expected_price_left);

        // Case 2: reserve0 == equilibrium_reserve0
        let price_eq = get_current_price(&params, eq_reserve_u256, eq_reserve_u256).unwrap();
        let expected_price_eq = mul_div_ceil(params.price_x, ONE_E18, params.price_y).unwrap();
        assert_eq!(price_eq, expected_price_eq);

        // Case 3: reserve0 > equilibrium_reserve0
        let reserve1_right = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_right =
            get_current_price(&params, eq_reserve_u256 + U256::from(1), reserve1_right).unwrap();
        let df_dx_right = df_dx(
            reserve1_right,
            params.price_y,
            params.price_x,
            eq_reserve_u256,
            params.concentration_y,
        )
        .unwrap();
        let expected_price_right = ceil_div(ONE_E36, df_dx_right).unwrap();
        assert_eq!(price_right, expected_price_right);

        // Case 4: reserve1 == equilibrium_reserve1 (on right side)
        let price_eq_right =
            get_current_price(&params, eq_reserve_u256 + U256::from(1), eq_reserve_u256).unwrap();
        let expected_price_eq_right =
            mul_div_ceil(params.price_y, ONE_E18, params.price_x).unwrap();
        assert_eq!(price_eq_right, expected_price_eq_right);
    }

    #[test]
    fn test_get_current_price_ray() {
        let eq_reserve_u256 = U256::from(100) * ONE_E18;
        let limbs = eq_reserve_u256.into_limbs();
        let eq_reserve_u112 = U112::from_limbs([limbs[0], limbs[1]]);
        let params = EulerSwapParams {
            vault0: Address::ZERO,
            vault1: Address::ZERO,
            euler_account: Address::ZERO,
            equilibrium_reserve0: eq_reserve_u112,
            equilibrium_reserve1: eq_reserve_u112,
            price_x: U256::from(2) * ONE_E18,
            price_y: U256::from(1) * ONE_E18,
            concentration_x: ONE_E18 / U256::from(10), // 0.1
            concentration_y: ONE_E18 / U256::from(5),  // 0.2
            fee: U256::ZERO,
            protocol_fee: U256::ZERO,
            protocol_fee_recipient: Address::ZERO,
        };

        // Case 1: reserve0 < equilibrium_reserve0
        let reserve0_left = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_left = get_current_price_ray(&params, reserve0_left, U256::ZERO).unwrap();
        let expected_price_left = df_dx_ray(
            reserve0_left,
            params.price_x,
            params.price_y,
            eq_reserve_u256,
            params.concentration_x,
        )
        .unwrap();
        assert_eq!(price_left, expected_price_left);

        // Case 2: reserve0 == equilibrium_reserve0
        let price_eq = get_current_price_ray(&params, eq_reserve_u256, eq_reserve_u256).unwrap();
        let expected_price_eq = mul_div_ceil(params.price_x, ONE_E27, params.price_y).unwrap();
        assert_eq!(price_eq, expected_price_eq);

        // Case 3: reserve0 > equilibrium_reserve0
        let reserve1_right = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_right =
            get_current_price_ray(&params, eq_reserve_u256 + U256::from(1), reserve1_right)
                .unwrap();
        let df_dx_right = df_dx_ray(
            reserve1_right,
            params.price_y,
            params.price_x,
            eq_reserve_u256,
            params.concentration_y,
        )
        .unwrap();
        let expected_price_right = ceil_div(ONE_E54, df_dx_right).unwrap();
        assert_eq!(price_right, expected_price_right);

        // Case 4: reserve1 == equilibrium_reserve1 (on right side)
        let price_eq_right =
            get_current_price_ray(&params, eq_reserve_u256 + U256::from(1), eq_reserve_u256)
                .unwrap();
        let expected_price_eq_right =
            mul_div_ceil(params.price_y, ONE_E27, params.price_x).unwrap();
        assert_eq!(price_eq_right, expected_price_eq_right);
    }

    #[test]
    fn test_curve_more_reserve0() {
        let params = EulerSwapParams {
            vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
            vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
            euler_account: address!("0x6fDBA16De9C131EF581069E02507c512A5574DbD"),
            equilibrium_reserve0: U112::from(7882338570209u128),
            equilibrium_reserve1: U112::from(2893638536189u128),
            price_x: U256::from(1000000u128),
            price_y: U256::from(1000472u128),
            concentration_x: U256::from(999000000000000100u128),
            concentration_y: U256::from(999000000000000100u128),
            fee: U256::from(50000000000000u128),
            protocol_fee: U256::from(0u128),
            protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
        };

        let reserve0 = U256::from(7222790500820u128);
        let reserve1 = U256::from(3552935643880u128);

        let is_valid = verify(&params, reserve0, reserve1).unwrap();
        assert!(is_valid, "The given reserves should be valid for the curve");

        // Test get_current_price
        let current_price = get_current_price(&params, reserve0, reserve1).unwrap();
        assert!(
            current_price > U256::ZERO,
            "Current price should be positive"
        );

        // Test get_current_price_ray
        let current_price_ray = get_current_price_ray(&params, reserve0, reserve1).unwrap();
        assert!(
            current_price_ray > U256::ZERO,
            "Current price in RAY should be positive"
        );

        // Test the roundtrip
        let (reserve0_roundtrip, reserve1_roundtrip) =
            get_current_reserves(&params, current_price).unwrap();
        assert_eq!(reserve0, reserve0_roundtrip);
        assert_eq!(reserve1, reserve1_roundtrip);

        // Test the roundtrip in ray
        let (reserve0_roundtrip_ray, reserve1_roundtrip_ray) =
            get_current_reserves_ray(&params, current_price_ray).unwrap();
        assert_eq!(reserve0, reserve0_roundtrip_ray);
        assert_eq!(reserve1, reserve1_roundtrip_ray);
    }

    #[test]
    fn test_curve_more_reserve1() {
        let params = EulerSwapParams {
            vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
            vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
            euler_account: address!("0x6fDBA16De9C131EF581069E02507c512A5574DbD"),
            equilibrium_reserve0: U112::from(7882338570209u128),
            equilibrium_reserve1: U112::from(2893638536189u128),
            price_x: U256::from(1000000u128),
            price_y: U256::from(1000472u128),
            concentration_x: U256::from(999000000000000100u128),
            concentration_y: U256::from(999000000000000100u128),
            fee: U256::from(50000000000000u128),
            protocol_fee: U256::from(0u128),
            protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
        };

        let reserve0 = U256::from(3552935643880u128);
        let reserve1 = U256::from(7226272020790u128);

        // Test verify function
        let is_valid = verify(&params, reserve0, reserve1).unwrap();
        assert!(is_valid, "The given reserves should be valid for the curve");

        // Test get_current_price
        let current_price = get_current_price(&params, reserve0, reserve1).unwrap();
        assert!(
            current_price > U256::ZERO,
            "Current price should be positive"
        );

        // Test get_current_price_ray
        let current_price_ray = get_current_price_ray(&params, reserve0, reserve1).unwrap();
        assert!(
            current_price_ray > U256::ZERO,
            "Current price in RAY should be positive"
        );

        // Test the roundtrip
        let (reserve0_roundtrip, reserve1_roundtrip) =
            get_current_reserves(&params, current_price).unwrap();
        assert_eq!(reserve0, reserve0_roundtrip);
        assert_eq!(reserve1, reserve1_roundtrip);

        // Test the roundtrip in ray
        let (reserve0_roundtrip_ray, reserve1_roundtrip_ray) =
            get_current_reserves_ray(&params, current_price_ray).unwrap();
        assert_eq!(reserve0, reserve0_roundtrip_ray);
        assert_eq!(reserve1, reserve1_roundtrip_ray);
    }

    #[test]
    fn test_get_current_reserves_1() {
        let params = EulerSwapParams {
            vault0: address!("0x6eae95ee783e4d862867c4e0e4c3f4b95aa682ba"),
            vault1: address!("0xd49181c522ecdb265f0d9c175cf26fface64ead3"),
            euler_account: address!("0x6fdba16de9c131ef581069e02507c512a5574dbd"),
            equilibrium_reserve0: uint!(7882338570209_U112),
            equilibrium_reserve1: uint!(2893638536189_U112),
            price_x: uint!(1000000_U256),
            price_y: uint!(1000472_U256),
            concentration_x: uint!(999000000000000100_U256),
            concentration_y: uint!(999000000000000100_U256),
            fee: uint!(50000000000000_U256),
            protocol_fee: uint!(0_U256),
            protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
        };

        let current_price = uint!(999714422196255613_U256);
        let (reserve0, reserve1) = get_current_reserves(&params, current_price).unwrap();
        assert_eq!(reserve0, uint!(7237025880666_U256));
        assert_eq!(reserve1, uint!(3538704296075_U256));
    }

    #[test]
    fn test_get_current_reserves_2() {
        let params = EulerSwapParams {
            concentration_x: uint!(690000000000000000_U256),
            concentration_y: uint!(900000000000000000_U256),
            equilibrium_reserve0: uint!(0_U112),
            equilibrium_reserve1: uint!(8924_U112),
            euler_account: address!("0x8cABEDFE7A52D73F8840e7eA9d50ad74cfCdFC8e"),
            fee: uint!(1000000000000000_U256),
            price_x: uint!(1000136039999999872_U256),
            price_y: uint!(999934470000000000_U256),
            protocol_fee: uint!(0_U256),
            protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
            vault0: address!("0x2888F098157162EC4a4274F7ad2c69921e95834D"),
            vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
        };

        // price too different
        let current_price = uint!(999719101195561546_U256);
        let result = get_current_reserves(&params, current_price);
        assert_eq!(result, Err(CurveError::NoSolution));
    }

    #[test]
    fn test_get_current_reserves_3() {
        let params = EulerSwapParams {
            vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
            vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
            euler_account: address!("0x6fDBA16De9C131EF581069E02507c512A5574DbD"),
            equilibrium_reserve0: U112::from(7882338570209u128),
            equilibrium_reserve1: U112::from(2893638536189u128),
            price_x: U256::from(1000000u128),
            price_y: U256::from(1000472u128),
            concentration_x: U256::from(999000000000000100u128),
            concentration_y: U256::from(999000000000000100u128),
            fee: U256::from(50000000000000u128),
            protocol_fee: U256::from(0u128),
            protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
        };

        // price too different
        let current_price = uint!(88684177661935620_U256);
        let result = get_current_reserves(&params, current_price);
        assert_eq!(result, Err(CurveError::NoSolution));
    }
}
