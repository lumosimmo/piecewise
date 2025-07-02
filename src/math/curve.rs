use crate::math::common::{
    MathError, ONE_E9, ONE_E18, ONE_E27, ONE_E36, ONE_E54, U248_MAX, ceil_div, delta_ratio,
    mul_div_ceil, sqrt_ceil,
};
use alloy::primitives::{Address, I256, U256, aliases::U112};
use alloy::sol;
use std::fmt;

sol! {
    /// The IEulerSwap.Params struct from the Solidity code.
    struct EulerSwapParams {
        address vault0;
        address vault1;
        address eulerAccount;
        uint112 equilibriumReserve0;
        uint112 equilibriumReserve1;
        uint256 priceX;
        uint256 priceY;
        uint256 concentrationX;
        uint256 concentrationY;
        uint256 fee;
        uint256 protocolFee;
        address protocolFeeRecipient;
    }
}

#[derive(Debug)]
pub enum CurveError {
    Math(MathError),
    PriceBelowApex,
    NoSolution,
}

impl From<MathError> for CurveError {
    fn from(e: MathError) -> Self {
        CurveError::Math(e)
    }
}

impl fmt::Debug for EulerSwapParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EulerSwapParams")
            .field("vault0", &self.vault0)
            .field("vault1", &self.vault1)
            .field("eulerAccount", &self.eulerAccount)
            .field("equilibriumReserve0", &self.equilibriumReserve0)
            .field("equilibriumReserve1", &self.equilibriumReserve1)
            .field("priceX", &self.priceX)
            .field("priceY", &self.priceY)
            .field("concentrationX", &self.concentrationX)
            .field("concentrationY", &self.concentrationY)
            .field("fee", &self.fee)
            .field("protocolFee", &self.protocolFee)
            .field("protocolFeeRecipient", &self.protocolFeeRecipient)
            .finish()
    }
}

/// Snake case getters for the EulerSwapParams struct because it looks better.
impl EulerSwapParams {
    pub fn vault0(&self) -> Address {
        self.vault0
    }

    pub fn vault1(&self) -> Address {
        self.vault1
    }

    pub fn euler_account(&self) -> Address {
        self.eulerAccount
    }

    pub fn equilibrium_reserve0(&self) -> U112 {
        self.equilibriumReserve0
    }

    pub fn equilibrium_reserve1(&self) -> U112 {
        self.equilibriumReserve1
    }

    pub fn price_x(&self) -> U256 {
        self.priceX
    }

    pub fn price_y(&self) -> U256 {
        self.priceY
    }

    pub fn concentration_x(&self) -> U256 {
        self.concentrationX
    }

    pub fn concentration_y(&self) -> U256 {
        self.concentrationY
    }

    pub fn fee(&self) -> U256 {
        self.fee
    }

    pub fn protocol_fee(&self) -> U256 {
        self.protocolFee
    }

    pub fn protocol_fee_recipient(&self) -> Address {
        self.protocolFeeRecipient
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
    let b: I256;
    let c_quad: U256;
    let four_ac: U256;

    let term1_unsigned = mul_div_ceil(py * ONE_E18, y - y0, px)?;
    let term1 = I256::from_raw(term1_unsigned); // scale: 1e36

    let i_c = I256::from_raw(c);
    let i_one_e18 = I256::from_raw(ONE_E18);
    let i_x0 = I256::from_raw(x0);
    let term2 = (I256::from(U256::from(2u64)) * i_c - i_one_e18) * i_x0; // scale: 1e36

    b = (term1 - term2) / i_one_e18; // scale: 1e18
    c_quad = mul_div_ceil(ONE_E18 - c, x0 * x0, ONE_E18)?; // scale: 1e36
    four_ac = mul_div_ceil(U256::from(4) * c, c_quad, ONE_E18)?; // scale: 1e36

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

    let x: U256;
    if b.is_negative() || b.is_zero() {
        x = mul_div_ceil(abs_b + sqrt, ONE_E18, U256::from(2) * c)? + U256::ONE;
    } else {
        x = ceil_div(U256::from(2) * c_quad, abs_b + sqrt)? + U256::ONE;
    }

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
    let bits = if x.is_zero() {
        0
    } else {
        x.log2() as usize + 1
    };

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

    let equilibrium_reserve0 = U256::from(p.equilibrium_reserve0());
    let equilibrium_reserve1 = U256::from(p.equilibrium_reserve1());

    if new_reserve0 >= equilibrium_reserve0 {
        if new_reserve1 >= equilibrium_reserve1 {
            return Ok(true);
        }
        Ok(new_reserve0
            >= f(
                new_reserve1,
                p.price_y(),
                p.price_x(),
                equilibrium_reserve1,
                equilibrium_reserve0,
                p.concentration_y(),
            )?)
    } else {
        if new_reserve1 < equilibrium_reserve1 {
            return Ok(false);
        }
        Ok(new_reserve1
            >= f(
                new_reserve0,
                p.price_x(),
                p.price_y(),
                equilibrium_reserve0,
                equilibrium_reserve1,
                p.concentration_x(),
            )?)
    }
}

/// Computes the marginal price at the given reserve vector, in WEI precision.
pub fn get_current_price(
    p: &EulerSwapParams,
    reserve0: U256,
    reserve1: U256,
) -> Result<U256, CurveError> {
    let equilibrium_reserve0 = U256::from(p.equilibrium_reserve0());
    let equilibrium_reserve1 = U256::from(p.equilibrium_reserve1());

    if reserve0 <= equilibrium_reserve0 {
        // We are on or to the left of the apex -> slope directly gives X‑price.
        if reserve0 == equilibrium_reserve0 {
            return Ok(mul_div_ceil(p.price_x(), ONE_E18, p.price_y())?);
        }
        df_dx(
            reserve0,
            p.price_x(),
            p.price_y(),
            equilibrium_reserve0,
            p.concentration_x(),
        )
    } else {
        // If on the right branch, derive the slope in Y‑space and invert.
        if reserve1 == equilibrium_reserve1 {
            return Ok(mul_div_ceil(p.price_y(), ONE_E18, p.price_x())?);
        }
        let price = df_dx(
            reserve1,
            p.price_y(),
            p.price_x(),
            equilibrium_reserve1,
            p.concentration_y(),
        )?;
        Ok(ceil_div(ONE_E36, price)?) // reciprocal because dx/dy = 1/(dy/dx)
    }
}

/// Computes the marginal price at the given reserve vector, in RAY precision.
pub fn get_current_price_ray(p: &EulerSwapParams, reserve0: U256, reserve1: U256) -> Result<U256, CurveError> {
    let equilibrium_reserve0 = U256::from(p.equilibrium_reserve0());
    let equilibrium_reserve1 = U256::from(p.equilibrium_reserve1());

    if reserve0 <= equilibrium_reserve0 {
        if reserve0 == equilibrium_reserve0 {
            return Ok(mul_div_ceil(p.price_x(), ONE_E27, p.price_y())?);
        }
        df_dx_ray(
            reserve0,
            p.price_x(),
            p.price_y(),
            equilibrium_reserve0,
            p.concentration_x(),
        )
    } else {
        if reserve1 == equilibrium_reserve1 {
            return Ok(mul_div_ceil(p.price_y(), ONE_E27, p.price_x())?);
        }
        let price = df_dx_ray(
            reserve1,
            p.price_y(),
            p.price_x(),
            equilibrium_reserve1,
            p.concentration_y(),
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
    let px = p.price_x();
    let py = p.price_y();
    let x0 = U256::from(p.equilibrium_reserve0());
    let y0 = U256::from(p.equilibrium_reserve1());
    let cx = p.concentration_x();
    let cy = p.concentration_y();

    let apex_price = mul_div_ceil(px, ONE_E18, py)?;
    if current_price < apex_price {
        return Err(CurveError::PriceBelowApex);
    }
    if current_price == apex_price {
        return Ok((x0, y0));
    }

    // Binary-search helper for monotonically *decreasing* function on [lo,hi].
    let search_left = || -> Result<Option<(U256, U256)>, CurveError> {
        let mut lo = U256::from(1);
        let mut hi = x0;
        while lo <= hi {
            let mid = (lo + hi) >> 1;
            let y = f(mid, px, py, x0, y0, cx)?;
            let price = get_current_price(p, mid, y)?;
            if price == current_price {
                return Ok(Some((mid, y)));
            }
            if price > current_price {
                lo = mid + U256::from(1);
            } else {
                hi = mid - U256::from(1);
            }
        }
        Ok(None)
    };

    // Binary-search helper for monotonically *increasing* function on [lo,hi].
    let search_right = || -> Result<Option<(U256, U256)>, CurveError> {
        let mut lo = y0;
        let mut hi = y0;

        loop {
            if hi.is_zero() {
                hi = U256::ONE;
            }
            let x_at_hi = f(hi, py, px, y0, x0, cy)?;
            let price = get_current_price(p, x_at_hi, hi)?;
            if price >= current_price {
                break;
            }
            hi <<= 1;
        }

        while lo <= hi {
            let mid: U256 = (lo + hi) >> 1;
            if mid.is_zero() {
                lo = U256::from(1);
                continue;
            }
            let x = f(mid, py, px, y0, x0, cy)?;
            let price = get_current_price(p, x, mid)?;
            if price == current_price {
                return Ok(Some((x, mid)));
            }
            if price < current_price {
                lo = mid + U256::from(1);
            } else {
                hi = mid - U256::from(1);
            }
        }
        Ok(None)
    };

    // Try left first (smaller numerical range – faster)
    if let Some(reserves) = search_left()? {
        return Ok(reserves);
    }
    if let Some(reserves) = search_right()? {
        return Ok(reserves);
    }
    Err(CurveError::NoSolution)
}

/// Finds the unique lattice point `(reserve0, reserve1)` whose marginal price equals to a given
/// price `current_price_ray`, in RAY precision. Binary search is used.
pub fn get_current_reserves_ray(
    p: &EulerSwapParams,
    current_price_ray: U256,
) -> Result<(U256, U256), CurveError> {
    let px = p.price_x();
    let py = p.price_y();
    let x0 = U256::from(p.equilibrium_reserve0());
    let y0 = U256::from(p.equilibrium_reserve1());
    let cx = p.concentration_x();
    let cy = p.concentration_y();

    let apex_price = mul_div_ceil(px, ONE_E27, py)?;
    if current_price_ray < apex_price {
        return Err(CurveError::PriceBelowApex);
    }
    if current_price_ray == apex_price {
        return Ok((x0, y0));
    }

    let search_left = || -> Result<Option<(U256, U256)>, CurveError> {
        let mut lo = U256::from(1);
        let mut hi = x0;
        while lo <= hi {
            let mid = (lo + hi) >> 1;
            let y = f(mid, px, py, x0, y0, cx)?;
            let price = get_current_price_ray(p, mid, y)?;
            // Sometimes we can't hit the exact price, so we have to exit when it's close enough.
            if delta_ratio(current_price_ray, price, ONE_E18)? == U256::ZERO {
                return Ok(Some((mid, y)));
            }
            if price > current_price_ray {
                lo = mid + U256::from(1);
            } else {
                hi = mid - U256::from(1);
            }
        }
        Ok(None)
    };

    // Binary-search helper for monotonically *increasing* function on [lo,hi].
    let search_right = || -> Result<Option<(U256, U256)>, CurveError> {
        let mut lo = y0;
        let mut hi = y0;

        loop {
            if hi.is_zero() {
                hi = U256::ONE;
            }
            let x_at_hi = f(hi, py, px, y0, x0, cy)?;
            let price = get_current_price_ray(p, x_at_hi, hi)?;
            if price > current_price_ray
                || delta_ratio(current_price_ray, price, ONE_E18)? == U256::ZERO
            {
                break;
            }
            hi <<= 1;
        }

        while lo <= hi {
            let mid: U256 = (lo + hi) >> 1;
            if mid.is_zero() {
                lo = U256::from(1);
                continue;
            }
            let x = f(mid, py, px, y0, x0, cy)?;
            let price = get_current_price_ray(p, x, mid)?;
            if delta_ratio(current_price_ray, price, ONE_E18)? == U256::ZERO {
                return Ok(Some((x, mid)));
            }
            if price < current_price_ray {
                lo = mid + U256::from(1);
            } else {
                hi = mid - U256::from(1);
            }
        }
        Ok(None)
    };

    if let Some(reserves) = search_left()? {
        return Ok(reserves);
    }
    if let Some(reserves) = search_right()? {
        return Ok(reserves);
    }
    Err(CurveError::NoSolution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::{ONE_E18, ONE_E27, ONE_E54};
    use alloy::primitives::{uint, U256};
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
            eulerAccount: Address::ZERO,
            equilibriumReserve0: eq_reserve_u112,
            equilibriumReserve1: eq_reserve_u112,
            priceX: ONE_E18,
            priceY: ONE_E18,
            concentrationX: ONE_E18,
            concentrationY: ONE_E18,
            fee: U256::ZERO,
            protocolFee: U256::ZERO,
            protocolFeeRecipient: Address::ZERO,
        };

        // Case 1: new reserves are above equilibrium
        assert!(verify(
            &params,
            eq_reserve_u256 + U256::from(1),
            eq_reserve_u256 + U256::from(1)
        )
        .unwrap());

        // Case 2: new_reserve0 >= eq0, new_reserve1 < eq1.
        // This case checks new_reserve0 >= f(new_reserve1, ...)
        let reserve0_case2 = U256::from(110) * ONE_E18;
        let reserve1_case2 = U256::from(90) * ONE_E18; // 110+90 = 200, so it's on the curve.
        assert!(verify(&params, reserve0_case2, reserve1_case2).unwrap());
        // Point below the curve
        assert!(!verify(
            &params,
            reserve0_case2 - U256::from(1),
            reserve1_case2
        )
        .unwrap());
        // Point above the curve
        assert!(verify(
            &params,
            reserve0_case2 + U256::from(1),
            reserve1_case2
        )
        .unwrap());

        // Case 3: new_reserve0 < eq0.
        // Must have new_reserve1 >= eq1 to proceed, otherwise false.
        assert!(!verify(
            &params,
            U256::from(90) * ONE_E18,
            U256::from(90) * ONE_E18
        )
        .unwrap());

        // This case checks new_reserve1 >= f(new_reserve0, ...)
        let reserve0_case3 = U256::from(90) * ONE_E18;
        let reserve1_case3 = U256::from(110) * ONE_E18; // 90+110 = 200, so it's on the curve.
        assert!(verify(&params, reserve0_case3, reserve1_case3).unwrap());
        // Point below the curve
        assert!(!verify(
            &params,
            reserve0_case3,
            reserve1_case3 - U256::from(1)
        )
        .unwrap());
        // Point above the curve
        assert!(verify(
            &params,
            reserve0_case3,
            reserve1_case3 + U256::from(1)
        )
        .unwrap());

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
            eulerAccount: Address::ZERO,
            equilibriumReserve0: eq_reserve_u112,
            equilibriumReserve1: eq_reserve_u112,
            priceX: U256::from(2) * ONE_E18,
            priceY: U256::from(1) * ONE_E18,
            concentrationX: ONE_E18 / U256::from(10), // 0.1
            concentrationY: ONE_E18 / U256::from(5),  // 0.2
            fee: U256::ZERO,
            protocolFee: U256::ZERO,
            protocolFeeRecipient: Address::ZERO,
        };

        // Case 1: reserve0 < equilibrium_reserve0
        let reserve0_left = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_left = get_current_price(&params, reserve0_left, U256::ZERO).unwrap();
        let expected_price_left = df_dx(
            reserve0_left,
            params.priceX,
            params.priceY,
            eq_reserve_u256,
            params.concentrationX,
        )
        .unwrap();
        assert_eq!(price_left, expected_price_left);

        // Case 2: reserve0 == equilibrium_reserve0
        let price_eq = get_current_price(&params, eq_reserve_u256, eq_reserve_u256).unwrap();
        let expected_price_eq = mul_div_ceil(params.priceX, ONE_E18, params.priceY).unwrap();
        assert_eq!(price_eq, expected_price_eq);

        // Case 3: reserve0 > equilibrium_reserve0
        let reserve1_right = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_right =
            get_current_price(&params, eq_reserve_u256 + U256::from(1), reserve1_right).unwrap();
        let df_dx_right = df_dx(
            reserve1_right,
            params.priceY,
            params.priceX,
            eq_reserve_u256,
            params.concentrationY,
        )
        .unwrap();
        let expected_price_right = ceil_div(ONE_E36, df_dx_right).unwrap();
        assert_eq!(price_right, expected_price_right);

        // Case 4: reserve1 == equilibrium_reserve1 (on right side)
        let price_eq_right =
            get_current_price(&params, eq_reserve_u256 + U256::from(1), eq_reserve_u256).unwrap();
        let expected_price_eq_right =
            mul_div_ceil(params.priceY, ONE_E18, params.priceX).unwrap();
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
            eulerAccount: Address::ZERO,
            equilibriumReserve0: eq_reserve_u112,
            equilibriumReserve1: eq_reserve_u112,
            priceX: U256::from(2) * ONE_E18,
            priceY: U256::from(1) * ONE_E18,
            concentrationX: ONE_E18 / U256::from(10), // 0.1
            concentrationY: ONE_E18 / U256::from(5),  // 0.2
            fee: U256::ZERO,
            protocolFee: U256::ZERO,
            protocolFeeRecipient: Address::ZERO,
        };

        // Case 1: reserve0 < equilibrium_reserve0
        let reserve0_left = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_left = get_current_price_ray(&params, reserve0_left, U256::ZERO).unwrap();
        let expected_price_left = df_dx_ray(
            reserve0_left,
            params.priceX,
            params.priceY,
            eq_reserve_u256,
            params.concentrationX,
        )
        .unwrap();
        assert_eq!(price_left, expected_price_left);

        // Case 2: reserve0 == equilibrium_reserve0
        let price_eq = get_current_price_ray(&params, eq_reserve_u256, eq_reserve_u256).unwrap();
        let expected_price_eq = mul_div_ceil(params.priceX, ONE_E27, params.priceY).unwrap();
        assert_eq!(price_eq, expected_price_eq);

        // Case 3: reserve0 > equilibrium_reserve0
        let reserve1_right = eq_reserve_u256 - (U256::from(10) * ONE_E18);
        let price_right =
            get_current_price_ray(&params, eq_reserve_u256 + U256::from(1), reserve1_right)
                .unwrap();
        let df_dx_right = df_dx_ray(
            reserve1_right,
            params.priceY,
            params.priceX,
            eq_reserve_u256,
            params.concentrationY,
        )
        .unwrap();
        let expected_price_right = ceil_div(ONE_E54, df_dx_right).unwrap();
        assert_eq!(price_right, expected_price_right);

        // Case 4: reserve1 == equilibrium_reserve1 (on right side)
        let price_eq_right =
            get_current_price_ray(&params, eq_reserve_u256 + U256::from(1), eq_reserve_u256)
                .unwrap();
        let expected_price_eq_right =
            mul_div_ceil(params.priceY, ONE_E27, params.priceX).unwrap();
        assert_eq!(price_eq_right, expected_price_eq_right);
    }
}
