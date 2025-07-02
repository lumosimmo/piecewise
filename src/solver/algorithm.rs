use alloy::primitives::U256;

use crate::math::common::{ONE_E36, ceil_div};
use crate::math::curve::{CurveError, get_current_price, get_current_reserves};
use crate::math::quote::find_curve_point;
use crate::solver::common::{
    ExactInSwapRequest, ExactInSwapResult, PoolSnapshot, RouteError, SwapAllocation,
};

/// Normalized per-pool state used during the greedy filling loop.
#[derive(Clone, Debug)]
struct RoutePool {
    /// Immutable starting snapshot (unchanged during simulation).
    snapshot: PoolSnapshot,

    /// Simulated reserves that mutate during the fill loop.
    reserve0: U256,
    reserve1: U256,

    /// Normalized marginal price – *output per input* – always scaled 18 decimals and
    /// **higher is better**, regardless of swap direction.
    price: U256,

    /// Cumulative input already assigned to this pool during the simulation.
    allocated_in: U256,

    /// `true` if `token0` is the input asset for this pool in this route.
    token0_is_input: bool,
}

impl RoutePool {
    /// Constructs a direction-normalized `RoutePool` from a raw snapshot and a `(token_in,token_out)` pair.
    fn try_new(
        snap: PoolSnapshot,
        token_in: &alloy::primitives::Address,
        token_out: &alloy::primitives::Address,
    ) -> Result<Self, RouteError> {
        // Determine if this pool can service the swap and in which orientation.
        let token0_is_input: bool;
        if &snap.token0 == token_in && &snap.token1 == token_out {
            token0_is_input = true;
        } else if &snap.token1 == token_in && &snap.token0 == token_out {
            token0_is_input = false;
        } else {
            // Wrong pair
            return Err(RouteError::NoViablePool);
        }

        let raw_price = get_current_price(&snap.params, snap.reserve0, snap.reserve1)?;

        let reserve0 = snap.reserve0;
        let reserve1 = snap.reserve1;

        // Normalize so that `price` is **out per in** and higher is better.
        let price = if token0_is_input {
            raw_price
        } else {
            ceil_div(ONE_E36, raw_price)?
        };

        Ok(Self {
            snapshot: snap,
            reserve0,
            reserve1,
            price,
            allocated_in: U256::ZERO,
            token0_is_input,
        })
    }
}

/// Greedy marginal-price equalization for finding the best route.
pub fn find_best_route_exact_in(
    pools: &[PoolSnapshot],
    request: &ExactInSwapRequest,
) -> Result<ExactInSwapResult, RouteError> {
    if request.amount_in.is_zero() {
        return Err(RouteError::AmountTooSmall);
    }
    if request.max_splits == 0 {
        return Err(RouteError::SplitsLimitTooLow);
    }

    let mut candidates: Vec<RoutePool> = pools
        .iter()
        .filter_map(|p| RoutePool::try_new(p.clone(), &request.token_in, &request.token_out).ok())
        .collect();

    if candidates.is_empty() {
        return Err(RouteError::NoViablePool);
    }

    candidates.sort_by(|a, b| b.price.cmp(&a.price));
    if candidates.len() > request.max_splits {
        candidates.truncate(request.max_splits);
    }

    let mut remaining_in = request.amount_in;

    while !remaining_in.is_zero() {
        let (best_idx, _) = candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.price.cmp(&b.price))
            .expect("[find_best_route_exact_in] candidate list should not be empty");
        let best_price = candidates[best_idx].price;

        // If only one pool or everyone tied, dump remainder into best and break
        let second_price_opt = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != best_idx)
            .map(|(_, p)| p.price)
            .max();
        if second_price_opt.is_none() || best_price == second_price_opt.unwrap() {
            candidates[best_idx].allocated_in += remaining_in;
            // Skip reserve updates and price recompute because we're exiting
            break;
        }

        let target_price = second_price_opt.unwrap();
        match compute_delta_to_target_price(&mut candidates[best_idx], target_price) {
            Ok(delta_in) => {
                if delta_in >= remaining_in {
                    candidates[best_idx].allocated_in += remaining_in;
                    remaining_in = U256::ZERO;
                } else {
                    candidates[best_idx].allocated_in += delta_in;
                    remaining_in -= delta_in;
                }
            }
            Err(e) => {
                // If the math fails, we pour all remaining input into the best pool and exit
                eprintln!(
                    "[find_best_route_exact_in] delta-solve failed: {:?}; falling back to single-pool fill",
                    e
                );
                candidates[best_idx].allocated_in += remaining_in;
                remaining_in = U256::ZERO;
            }
        }
    }

    let mut allocations: Vec<SwapAllocation> = Vec::new();
    let mut total_out = U256::ZERO;

    for p in candidates.into_iter() {
        if p.allocated_in.is_zero() {
            continue;
        }
        let out = find_curve_point(
            &p.snapshot.params,
            p.allocated_in,
            true,
            p.token0_is_input,
            p.snapshot.reserve0,
            p.snapshot.reserve1,
        )?;
        total_out += out;

        allocations.push(SwapAllocation {
            pool: p.snapshot.pool_address,
            amount_in: p.allocated_in,
        });
    }

    Ok(ExactInSwapResult {
        total_out,
        allocations,
    })
}

/// Computes the input amount needed to lower a pool's *normalized* marginal price to `target_price`.
fn compute_delta_to_target_price(
    pool: &mut RoutePool,
    target_price: U256,
) -> Result<U256, RouteError> {
    use crate::math::common::{ONE_E36, ceil_div};

    if target_price >= pool.price {
        return Ok(U256::ZERO); // already at or below target
    }

    if pool.token0_is_input {
        match get_current_reserves(&pool.snapshot.params, target_price) {
            Ok((new_reserve0, new_reserve1)) => {
                if new_reserve0 <= pool.reserve0 {
                    return Err(RouteError::ComputationFailed(CurveError::NoSolution));
                }

                let delta_in = new_reserve0 - pool.reserve0;

                pool.reserve0 = new_reserve0;
                pool.reserve1 = new_reserve1;
                pool.price = target_price;

                Ok(delta_in)
            }
            Err(e) => Err(RouteError::ComputationFailed(e)),
        }
    } else {
        let inverse_target = ceil_div(ONE_E36, target_price)?;

        match get_current_reserves(&pool.snapshot.params, inverse_target) {
            Ok((new_reserve0, new_reserve1)) => {
                if new_reserve1 <= pool.reserve1 {
                    return Err(RouteError::ComputationFailed(CurveError::NoSolution));
                }

                let delta_in = new_reserve1 - pool.reserve1;

                pool.reserve0 = new_reserve0;
                pool.reserve1 = new_reserve1;
                pool.price = target_price;

                Ok(delta_in)
            }
            Err(e) => Err(RouteError::ComputationFailed(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::ONE_E18;
    use crate::math::curve::{EulerSwapParams, get_current_price};
    use alloy::primitives::{Address, aliases::U112};

    fn create_test_params() -> EulerSwapParams {
        let eq_reserve = U256::from(100) * ONE_E18;
        let limbs = eq_reserve.into_limbs();
        let eq_reserve_u112 = U112::from_limbs([limbs[0], limbs[1]]);

        EulerSwapParams {
            vault0: Address::ZERO,
            vault1: Address::ZERO,
            eulerAccount: Address::ZERO,
            equilibriumReserve0: eq_reserve_u112,
            equilibriumReserve1: eq_reserve_u112,
            priceX: ONE_E18,
            priceY: ONE_E18,
            concentrationX: ONE_E18 / U256::from(2), // 0.5
            concentrationY: ONE_E18 / U256::from(2), // 0.5
            fee: U256::ZERO,
            protocolFee: U256::ZERO,
            protocolFeeRecipient: Address::ZERO,
        }
    }

    #[test]
    fn test_compute_delta_to_target_price() {
        let params = create_test_params();
        let snapshot = PoolSnapshot {
            pool_address: Address::ZERO,
            token0: Address::ZERO,
            token1: Address::ZERO,
            reserve0: U256::from(80) * ONE_E18,
            reserve1: U256::from(120) * ONE_E18,
            params: params.clone(),
        };

        let mut pool_state = RoutePool {
            snapshot,
            reserve0: U256::from(80) * ONE_E18,
            reserve1: U256::from(120) * ONE_E18,
            price: get_current_price(&params, U256::from(80) * ONE_E18, U256::from(120) * ONE_E18)
                .unwrap(),
            allocated_in: U256::ZERO,
            token0_is_input: true,
        };
        let initial_price = pool_state.price;

        // Target a slightly lower price (which requires adding token0)
        let target_price = initial_price - (initial_price / U256::from(10)); // 90% of current

        match compute_delta_to_target_price(&mut pool_state, target_price) {
            Ok(delta) => {
                assert!(delta > U256::ZERO);
                assert_eq!(pool_state.price, target_price);
                assert!(pool_state.reserve0 > U256::from(80) * ONE_E18);
            }
            Err(_) => {
                panic!("[test_compute_delta_to_target_price] compute_delta_to_target_price failed");
            }
        }
    }

    #[test]
    fn test_find_best_route_single_pool() {
        let params = create_test_params();
        let pools = vec![PoolSnapshot {
            pool_address: Address::ZERO,
            token0: Address::from([1u8; 20]),
            token1: Address::from([2u8; 20]),
            reserve0: U256::from(80) * ONE_E18,
            reserve1: U256::from(120) * ONE_E18,
            params,
        }];

        let request = ExactInSwapRequest {
            token_in: Address::from([1u8; 20]),
            token_out: Address::from([2u8; 20]),
            amount_in: U256::from(10) * ONE_E18,
            max_splits: 1,
        };

        let result = find_best_route_exact_in(&pools, &request).unwrap();

        assert_eq!(result.allocations.len(), 1);
        assert_eq!(result.allocations[0].amount_in, U256::from(10) * ONE_E18);
        assert!(result.total_out > U256::ZERO);
    }
}
