use crate::math::curve::{CurveError, EulerSwapParams, f, f_inverse};
use alloy::primitives::U256;

/// Calculates the output amount for a swap, given an input amount and the pool state.
pub fn find_curve_point(
    p: &EulerSwapParams,
    amount: U256,
    exact_in: bool,
    token0_is_input: bool,
    reserve0: U256,
    reserve1: U256,
) -> Result<U256, CurveError> {
    let px = p.price_x();
    let py = p.price_y();
    let x0 = U256::from(p.equilibrium_reserve0());
    let y0 = U256::from(p.equilibrium_reserve1());
    let cx = p.concentration_x();
    let cy = p.concentration_y();

    if exact_in {
        if token0_is_input {
            // exact in, swap X in and Y out
            let x_new = reserve0 + amount;
            let y_new = if x_new <= x0 {
                f(x_new, px, py, x0, y0, cx)?
            } else {
                f_inverse(x_new, py, px, y0, x0, cy)?
            };
            Ok(reserve1.saturating_sub(y_new))
        } else {
            // exact in, swap Y in and X out
            let y_new = reserve1 + amount;
            let x_new = if y_new <= y0 {
                f(y_new, py, px, y0, x0, cy)?
            } else {
                f_inverse(y_new, px, py, x0, y0, cx)?
            };
            Ok(reserve0.saturating_sub(x_new))
        }
    } else {
        // exact out
        if token0_is_input {
            // exact out, swap Y out and X in
            let y_new = reserve1
                .checked_sub(amount)
                .expect("[find_curve_point] swap limit exceeded");
            let x_new = if y_new <= y0 {
                f(y_new, py, px, y0, x0, cy)?
            } else {
                f_inverse(y_new, px, py, x0, y0, cx)?
            };
            Ok(x_new.saturating_sub(reserve0))
        } else {
            // exact out, swap X out and Y in
            let x_new = reserve0
                .checked_sub(amount)
                .expect("[find_curve_point] swap limit exceeded");
            let y_new = if x_new <= x0 {
                f(x_new, px, py, x0, y0, cx)?
            } else {
                f_inverse(x_new, py, px, y0, x0, cy)?
            };
            Ok(y_new.saturating_sub(reserve1))
        }
    }
}
