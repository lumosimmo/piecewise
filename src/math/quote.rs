use crate::math::common::ONE_E18;
use crate::math::curve::{CurveError, EulerSwapParams, f, f_inverse};
use alloy_primitives::{U256, aliases::U112};

/// Computes the output amount for a swap, given an input amount and the pool state.
pub fn find_curve_point(
    p: &EulerSwapParams,
    reserve0: U256,
    reserve1: U256,
    amount: U256,
    exact_in: bool,
    token0_is_input: bool,
) -> Result<U256, CurveError> {
    let price_x = p.price_x;
    let price_y = p.price_y;
    let x0 = U256::from(p.equilibrium_reserve0);
    let y0 = U256::from(p.equilibrium_reserve1);
    let concentration_x = p.concentration_x;
    let concentration_y = p.concentration_y;

    if exact_in {
        if token0_is_input {
            // Exact in, swap X for Y
            let x_new = reserve0 + amount;
            let y_new = if x_new <= x0 {
                f(x_new, price_x, price_y, x0, y0, concentration_x)?
            } else {
                f_inverse(x_new, price_y, price_x, y0, x0, concentration_y)?
            };
            Ok(reserve1.saturating_sub(y_new))
        } else {
            // Exact in, swap Y for X
            let y_new = reserve1 + amount;
            let x_new = if y_new <= y0 {
                f(y_new, price_y, price_x, y0, x0, concentration_y)?
            } else {
                f_inverse(y_new, price_x, price_y, x0, y0, concentration_x)?
            };
            Ok(reserve0.saturating_sub(x_new))
        }
    } else {
        if token0_is_input {
            // Exact out, swap Y out for X in
            if reserve1 < amount {
                return Err(CurveError::SwapLimitExceeded);
            }
            let y_new = reserve1 - amount;
            let x_new = if y_new <= y0 {
                f(y_new, price_y, price_x, y0, x0, concentration_y)?
            } else {
                f_inverse(y_new, price_x, price_y, x0, y0, concentration_x)?
            };
            Ok(x_new.saturating_sub(reserve0))
        } else {
            // Exact out, swap X out for Y in
            if reserve0 < amount {
                return Err(CurveError::SwapLimitExceeded);
            }
            let x_new = reserve0 - amount;
            let y_new = if x_new <= x0 {
                f(x_new, price_x, price_y, x0, y0, concentration_x)?
            } else {
                f_inverse(x_new, price_y, price_x, y0, x0, concentration_y)?
            };
            Ok(y_new.saturating_sub(reserve1))
        }
    }
}

/// Computes the reserve1, given the pool params and the reserve0.
pub fn find_reserve1_for_reserve0(p: &EulerSwapParams, reserve0: U256) -> Result<U256, CurveError> {
    let price_x = p.price_x;
    let price_y = p.price_y;
    let x0 = U256::from(p.equilibrium_reserve0);
    let y0 = U256::from(p.equilibrium_reserve1);
    let concentration_x = p.concentration_x;
    let concentration_y = p.concentration_y;

    let reserve1 = if reserve0 <= x0 {
        f(reserve0, price_x, price_y, x0, y0, concentration_x)?
    } else {
        f_inverse(reserve0, price_y, price_x, y0, x0, concentration_y)?
    };
    Ok(reserve1)
}

/// Computes the reserve0, given the pool params and the reserve1.
pub fn find_reserve0_for_reserve1(p: &EulerSwapParams, reserve1: U256) -> Result<U256, CurveError> {
    let price_x = p.price_x;
    let price_y = p.price_y;
    let x0 = U256::from(p.equilibrium_reserve0);
    let y0 = U256::from(p.equilibrium_reserve1);
    let concentration_x = p.concentration_x;
    let concentration_y = p.concentration_y;

    let reserve0 = if reserve1 <= y0 {
        f(reserve1, price_y, price_x, y0, x0, concentration_y)?
    } else {
        f_inverse(reserve1, price_x, price_y, x0, y0, concentration_x)?
    };
    Ok(reserve0)
}

/// Computes the output amount for a swap, given an input amount and the pool state. Fee is included.
pub fn compute_quote(
    p: &EulerSwapParams,
    reserve0: U256,
    reserve1: U256,
    amount: U256,
    exact_in: bool,
    token0_is_input: bool,
) -> Result<U256, CurveError> {
    if amount.is_zero() {
        return Ok(U256::ZERO);
    }
    if amount > U256::from(U112::MAX) {
        return Err(CurveError::SwapLimitExceeded);
    }

    let fee = p.fee;
    let amount_for_curve = if exact_in {
        amount
    } else {
        amount.saturating_sub(amount * fee / ONE_E18)
    };

    let mut quote = find_curve_point(
        p,
        reserve0,
        reserve1,
        amount_for_curve,
        exact_in,
        token0_is_input,
    )?;

    if !exact_in {
        quote = quote * ONE_E18 / (ONE_E18 - fee);
    }

    Ok(quote)
}
