use alloy_primitives::U256;
use std::collections::HashSet;

use crate::math::curve::{get_current_price, get_current_reserves};
use crate::math::quote::{compute_quote, find_reserve0_for_reserve1, find_reserve1_for_reserve0};
use crate::solver::common::{
    ExactInSwapRequest, ExactInSwapResult, PoolSnapshot, RouteError, SwapAllocation,
};
#[cfg(target_arch = "wasm32")]
use serde_wasm_bindgen::{Error, from_value, to_value};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Per-pool state used during the greedy filling loop.
#[derive(Clone, Debug)]
struct RoutePool {
    /// Immutable starting snapshot (unchanged during simulation).
    snapshot: PoolSnapshot,

    /// Simulated reserves that mutate during the fill loop.
    reserve0: U256,
    reserve1: U256,

    /// Marginal price of the pool. When token0 is input, higher is better. When token0 is output, lower is better.
    price: U256,

    /// Cumulative input already assigned to this pool during the simulation.
    allocated_in: U256,

    /// `true` if `token0` is the input asset for this pool in this route.
    token0_is_input: bool,
}

impl RoutePool {
    /// Constructs a `RoutePool` from a raw snapshot and a `(token_in,token_out)` pair.
    fn try_new(
        snap: PoolSnapshot,
        token_in: &alloy_primitives::Address,
        token_out: &alloy_primitives::Address,
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

        let price = get_current_price(&snap.params, snap.reserve0, snap.reserve1)?;

        let reserve0 = snap.reserve0;
        let reserve1 = snap.reserve1;

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

/// Greedy marginal-price equalization and chunk-filling for finding the best route.
pub fn find_best_route_exact_in(
    pools: &[PoolSnapshot],
    request: &ExactInSwapRequest,
) -> Result<ExactInSwapResult, RouteError> {
    if request.amount_in.is_zero() {
        return Err(RouteError::AmountTooSmall);
    }

    let mut candidates: Vec<RoutePool> = pools
        .iter()
        .filter_map(|p| RoutePool::try_new(p.clone(), &request.token_in, &request.token_out).ok())
        .collect();

    candidates.retain(|pool| {
        if pool.token0_is_input {
            !pool.reserve1.is_zero()
        } else {
            !pool.reserve0.is_zero()
        }
    });

    if candidates.is_empty() {
        return Err(RouteError::NoViablePool);
    }

    // Sort candidates by best marginal price
    // When token0 is input, we want lower prices (cheaper to buy token1) and vice versa
    candidates.sort_by(|a, b| {
        if a.token0_is_input {
            b.price.cmp(&a.price)
        } else {
            a.price.cmp(&b.price)
        }
    });

    let mut remaining_in = request.amount_in;
    let mut equalized_pools: usize = 1;
    let mut dead_pools: HashSet<usize> = HashSet::new();
    let mut liquid_pools: HashSet<usize> = HashSet::new();

    // Marginal-price equalization
    while equalized_pools < candidates.len() && remaining_in > U256::ZERO {
        for i in 0..equalized_pools {
            if dead_pools.contains(&i) || dead_pools.contains(&equalized_pools) {
                continue;
            }

            // If result is Err NoSolution, we just treat it as 0.
            let target_price = candidates[equalized_pools].price;
            let delta = match compute_delta_to_target_price(&candidates[i], target_price) {
                Ok(delta) => {
                    liquid_pools.insert(i);
                    liquid_pools.insert(equalized_pools);
                    delta
                }
                Err(_) => {
                    if liquid_pools.contains(&i) {
                        dead_pools.insert(equalized_pools);
                    } else {
                        dead_pools.insert(i);
                    }
                    U256::ZERO
                }
            };
            if delta.is_zero() {
                continue;
            }

            if remaining_in > delta {
                remaining_in -= delta;
                candidates[i].allocated_in += delta;
                if candidates[i].token0_is_input {
                    candidates[i].reserve0 += delta;
                    candidates[i].reserve1 = find_reserve1_for_reserve0(
                        &candidates[i].snapshot.params,
                        candidates[i].reserve0,
                    )?;
                } else {
                    candidates[i].reserve1 += delta;
                    candidates[i].reserve0 = find_reserve0_for_reserve1(
                        &candidates[i].snapshot.params,
                        candidates[i].reserve1,
                    )?;
                }
            } else {
                candidates[i].allocated_in += remaining_in;
                if candidates[i].token0_is_input {
                    candidates[i].reserve0 += remaining_in;
                    candidates[i].reserve1 = find_reserve1_for_reserve0(
                        &candidates[i].snapshot.params,
                        candidates[i].reserve0,
                    )?;
                } else {
                    candidates[i].reserve1 += remaining_in;
                    candidates[i].reserve0 = find_reserve0_for_reserve1(
                        &candidates[i].snapshot.params,
                        candidates[i].reserve1,
                    )?;
                }
                remaining_in = U256::ZERO;
            }
            candidates[i].price = get_current_price(
                &candidates[i].snapshot.params,
                candidates[i].reserve0,
                candidates[i].reserve1,
            )?;
        }
        equalized_pools += 1;
    }

    // Chunk-filling if there is remaining input
    if remaining_in > U256::ZERO {
        // Sort candidates by biggest output reserve
        // When token0 is input, we want higher reserve1 and vice versa
        candidates.sort_by(|a, b| {
            if a.token0_is_input {
                b.reserve1.cmp(&a.reserve1)
            } else {
                a.reserve1.cmp(&b.reserve1)
            }
        });

        // Distribute remaining input in 1% chunks
        let one_percent = remaining_in / U256::from(100);
        let min_chunk = if one_percent.is_zero() {
            remaining_in
        } else {
            one_percent
        };

        while remaining_in > U256::ZERO {
            let chunk = if remaining_in >= min_chunk {
                min_chunk
            } else {
                remaining_in
            };

            let mut max_out = U256::ZERO;
            let mut max_out_pool = 0;
            for i in 0..candidates.len() {
                let out = compute_quote(
                    &candidates[i].snapshot.params,
                    candidates[i].reserve0 + chunk,
                    candidates[i].reserve1,
                    chunk,
                    true,
                    candidates[i].token0_is_input,
                )?;
                if out > max_out {
                    max_out = out;
                    max_out_pool = i;
                }
            }

            candidates[max_out_pool].allocated_in += chunk;
            remaining_in -= chunk;
            if candidates[max_out_pool].token0_is_input {
                candidates[max_out_pool].reserve0 += chunk;
                candidates[max_out_pool].reserve1 = find_reserve1_for_reserve0(
                    &candidates[max_out_pool].snapshot.params,
                    candidates[max_out_pool].reserve0,
                )?;
            } else {
                candidates[max_out_pool].reserve1 += chunk;
                candidates[max_out_pool].reserve0 = find_reserve0_for_reserve1(
                    &candidates[max_out_pool].snapshot.params,
                    candidates[max_out_pool].reserve1,
                )?;
            }
            candidates[max_out_pool].price = get_current_price(
                &candidates[max_out_pool].snapshot.params,
                candidates[max_out_pool].reserve0,
                candidates[max_out_pool].reserve1,
            )?;
        }
    }

    let mut allocations: Vec<SwapAllocation> = Vec::new();
    let mut total_out = U256::ZERO;

    for p in candidates.into_iter() {
        if p.allocated_in.is_zero() {
            continue;
        }
        let out = compute_quote(
            &p.snapshot.params,
            p.snapshot.reserve0,
            p.snapshot.reserve1,
            p.allocated_in,
            true,
            p.token0_is_input,
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

/// Computes the input amount needed to improve a pool's marginal price to `target_price`.
fn compute_delta_to_target_price(pool: &RoutePool, target_price: U256) -> Result<U256, RouteError> {
    if pool.token0_is_input {
        if target_price >= pool.price {
            return Ok(U256::ZERO);
        }
    } else {
        if target_price <= pool.price {
            return Ok(U256::ZERO);
        }
    }

    let (reserve0, reserve1) = get_current_reserves(&pool.snapshot.params, target_price)?;

    let delta_in = if pool.token0_is_input {
        reserve0 - pool.reserve0
    } else {
        reserve1 - pool.reserve1
    };

    Ok(delta_in)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::ONE_E6;
    use crate::math::curve::EulerSwapParams;
    use alloy_primitives::{address, uint};

    fn create_test_pools() -> Vec<PoolSnapshot> {
        vec![
            PoolSnapshot {
                pool_address: address!("0x97711bc4e7Ebc1b1d691D54F3769A23544D9a8a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x9151434b16b9763660705744891fa906f660ecc5"),
                reserve0: uint!(7222788990895_U256),
                reserve1: uint!(3552937153382_U256),
                params: EulerSwapParams {
                    concentration_x: uint!(999000000000000100_U256),
                    concentration_y: uint!(999000000000000100_U256),
                    equilibrium_reserve0: uint!(7882338570209_U112),
                    equilibrium_reserve1: uint!(2893638536189_U112),
                    euler_account: address!("0x6fDBA16De9C131EF581069E02507c512A5574DbD"),
                    fee: uint!(50000000000000_U256),
                    price_x: uint!(1000000_U256),
                    price_y: uint!(1000472_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x96C68a406187f8C86DEA6b3c6150ad5A176128A8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x9151434b16b9763660705744891fa906f660ecc5"),
                reserve0: uint!(7517178463564_U256),
                reserve1: uint!(28735470974821_U256),
                params: EulerSwapParams {
                    concentration_x: uint!(999950000000000000_U256),
                    concentration_y: uint!(999950000000000000_U256),
                    equilibrium_reserve0: uint!(18725660339732_U112),
                    equilibrium_reserve1: uint!(17532270426334_U112),
                    euler_account: address!("0x6fDBa16dE9C131eF581069E02507c512a5574DBf"),
                    fee: uint!(10000000000000_U256),
                    price_x: uint!(1000000_U256),
                    price_y: uint!(1000546_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x83A04dE9f9BdcE80E5F83d7fB830741daA2D28a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x9151434b16b9763660705744891fa906f660ecc5"),
                reserve0: uint!(0_U256),
                reserve1: uint!(8924_U256),
                params: EulerSwapParams {
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
                },
            },
            PoolSnapshot {
                pool_address: address!("0x7615160EDf1b4e2791A88C1012F0fA83e51B28a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x9151434b16b9763660705744891fa906f660ecc5"),
                reserve0: uint!(1930394388_U256),
                reserve1: uint!(35252595_U256),
                params: EulerSwapParams {
                    concentration_x: uint!(900000000000000000_U256),
                    concentration_y: uint!(900000000000000000_U256),
                    equilibrium_reserve0: uint!(1309048315_U112),
                    equilibrium_reserve1: uint!(359033607_U112),
                    euler_account: address!("0x7eA194FF26a2264d5C58e94C8aaC76569c999741"),
                    fee: uint!(10000000000000_U256),
                    price_x: uint!(1000000_U256),
                    price_y: uint!(1000298_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0xD49181c522eCDB265f0D9C175Cf26FFACE64eAD3"),
                },
            },
        ]
    }

    #[test]
    fn test_find_best_route_exact_in() {
        let pools = create_test_pools();

        let request = ExactInSwapRequest {
            // USDT
            token_in: address!("0x9151434b16b9763660705744891fa906f660ecc5"),
            // USDC
            token_out: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
            amount_in: U256::from(10000000) * ONE_E6,
        };

        let result = find_best_route_exact_in(&pools, &request).unwrap();

        assert_eq!(result.allocations.len(), 3);
        assert!(result.total_out > U256::from(9900000) * ONE_E6);
        assert!(result.total_out < U256::from(10100000) * ONE_E6);
    }

    fn create_test_pools_2() -> Vec<PoolSnapshot> {
        vec![
            PoolSnapshot {
                pool_address: address!("0x38d285ec702d344467715ff1F482591B01AD68a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(1001459_U256),
                reserve1: uint!(0_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0xDBd5c529E84f2a1955171578E1DB86eFF56A2884"),
                    equilibrium_reserve0: uint!(1001459_U112),
                    equilibrium_reserve1: uint!(0_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2761581097_U256),
                    concentration_x: uint!(0_U256),
                    concentration_y: uint!(0_U256),
                    fee: uint!(500000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0xf467191a3ED4302342d3AD133D4744C81Ea228A8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(0_U256),
                reserve1: uint!(400000003163559_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x694A225ee37B5DD67F3f15BCa7472C1Ee1918A14"),
                    equilibrium_reserve0: uint!(0_U112),
                    equilibrium_reserve1: uint!(400000003163559_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2575564870_U256),
                    concentration_x: uint!(900000000000000000_U256),
                    concentration_y: uint!(990000000000000000_U256),
                    fee: uint!(1000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x5EC4eDf777c15655b41FB9EFd5F90541d873e8a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(12759775_U256),
                reserve1: uint!(68627772442675472_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x8476B695C5F8313CbB6B46b5B18dA11AEd1b6BDb"),
                    equilibrium_reserve0: uint!(188940518_U112),
                    equilibrium_reserve1: uint!(0_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2568965422_U256),
                    concentration_x: uint!(999950000000000000_U256),
                    concentration_y: uint!(999950000000000000_U256),
                    fee: uint!(10000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x903DC95DB13e221456DDC3Fb6119B8d7cC68E8a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(786028365_U256),
                reserve1: uint!(288055812799891228_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x75102a2309cD305c7457a72397D0BcC000C4e040"),
                    equilibrium_reserve0: uint!(786028365_U112),
                    equilibrium_reserve1: uint!(288055812799891228_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2428441526_U256),
                    concentration_x: uint!(0_U256),
                    concentration_y: uint!(0_U256),
                    fee: uint!(500000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x0802d84c163ffF54c133873eB045d7DA6CDA68A8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(802297567_U256),
                reserve1: uint!(281582336437558876_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x75102a2309CD305c7457a72397D0Bcc000c4e042"),
                    equilibrium_reserve0: uint!(788618385_U112),
                    equilibrium_reserve1: uint!(287063727791972180_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2447915334_U256),
                    concentration_x: uint!(0_U256),
                    concentration_y: uint!(0_U256),
                    fee: uint!(500000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x7e7eeFa399cEd60930aa0c55d13e58bFA9FB68a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(76985989_U256),
                reserve1: uint!(19863016048953716_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x8476b695C5F8313Cbb6b46B5b18da11aed1B6BDd"),
                    equilibrium_reserve0: uint!(67500026_U112),
                    equilibrium_reserve1: uint!(23698581619701112_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2426306515_U256),
                    concentration_x: uint!(900000000000000000_U256),
                    concentration_y: uint!(900000000000000000_U256),
                    fee: uint!(1000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x52317f38F19cD8cF0100A6268791877A16D828A8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(779879390_U256),
                reserve1: uint!(194679249285341940_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x8476B695C5f8313CBB6b46b5b18DA11AED1B6BD9"),
                    equilibrium_reserve0: uint!(675000121_U112),
                    equilibrium_reserve1: uint!(236985764060152467_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2426306515_U256),
                    concentration_x: uint!(900000000000000000_U256),
                    concentration_y: uint!(900000000000000000_U256),
                    fee: uint!(1000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x0dAA7a2eb668131E1B353Aaa4cb2E0CF6B66E8A8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(2000000000_U256),
                reserve1: uint!(2000000000_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x29d5ea019FA72B489C44F15b7E95771b399D37Ef"),
                    equilibrium_reserve0: uint!(2000000000_U112),
                    equilibrium_reserve1: uint!(2000000000_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(1000000000000000000_U256),
                    concentration_x: uint!(970000000000000000_U256),
                    concentration_y: uint!(970000000000000000_U256),
                    fee: uint!(10000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0xec078526b7a841c3f8FCD13ECC8EfC0F1E25A8a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(50317008_U256),
                reserve1: uint!(15629487686265722_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x3e4Ab57CD1Aef2DE5D2d3b889d6f2Fc82b5Dc733"),
                    equilibrium_reserve0: uint!(43361827_U112),
                    equilibrium_reserve1: uint!(18427910303646597_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2441675542_U256),
                    concentration_x: uint!(900000000000000000_U256),
                    concentration_y: uint!(900000000000000000_U256),
                    fee: uint!(1000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0x8Dbe9d4b9318B79c94D8C3C541e28d2cd782E8A8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(7973780_U256),
                reserve1: uint!(1878373679314331_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x8476b695C5F8313cBB6b46B5B18Da11AED1b6BD8"),
                    equilibrium_reserve0: uint!(6750046_U112),
                    equilibrium_reserve1: uint!(2369873803027723_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2426306515_U256),
                    concentration_x: uint!(900000000000000000_U256),
                    concentration_y: uint!(900000000000000000_U256),
                    fee: uint!(1000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
            PoolSnapshot {
                pool_address: address!("0xcF9A7291E051D3f3070dA681c0552Fc41BFC68a8"),
                token0: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
                token1: address!("0x4200000000000000000000000000000000000006"),
                reserve0: uint!(5337332_U256),
                reserve1: uint!(2584091935708467_U256),
                params: EulerSwapParams {
                    vault0: address!("0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"),
                    vault1: address!("0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"),
                    euler_account: address!("0x75102a2309cD305c7457a72397D0BCc000c4E046"),
                    equilibrium_reserve0: uint!(5337332_U112),
                    equilibrium_reserve1: uint!(2584091935708467_U112),
                    price_x: uint!(1000000000000000000_U256),
                    price_y: uint!(2424667625_U256),
                    concentration_x: uint!(0_U256),
                    concentration_y: uint!(0_U256),
                    fee: uint!(1000000000000000_U256),
                    protocol_fee: uint!(0_U256),
                    protocol_fee_recipient: address!("0x0000000000000000000000000000000000000000"),
                },
            },
        ]
    }

    #[test]
    fn test_find_best_route_exact_in_2() {
        let pools = create_test_pools_2();

        let request = ExactInSwapRequest {
            // USDC
            token_in: address!("0x078d782b760474a361dda0af3839290b0ef57ad6"),
            // WETH
            token_out: address!("0x4200000000000000000000000000000000000006"),
            amount_in: U256::from(1) * ONE_E6,
        };

        let result = find_best_route_exact_in(&pools, &request).unwrap();

        assert_eq!(result.allocations.len(), 2);
        assert!(result.total_out > uint!(411000000000000_U256));
        assert!(result.total_out < uint!(412000000000000_U256));
    }
}

// WASM wrapper functions
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_find_best_route_exact_in(pools: JsValue, request: JsValue) -> Result<JsValue, Error> {
    let pools: Vec<PoolSnapshot> = from_value(pools)?;
    let request: ExactInSwapRequest = from_value(request)?;

    let result = find_best_route_exact_in(&pools, &request)?;
    to_value(&result)
}
