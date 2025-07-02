use alloy::primitives::{Address, U256};

use crate::math::common::MathError;
use crate::math::curve::{CurveError, EulerSwapParams};

#[derive(Clone, Debug)]
pub struct PoolSnapshot {
    pub pool_address: Address,
    pub token0: Address,
    pub token1: Address,
    pub reserve0: U256,
    pub reserve1: U256,
    pub params: EulerSwapParams,
}

#[derive(Clone, Debug)]
pub struct ExactInSwapRequest {
    pub token_in: Address,
    pub token_out: Address,
    pub amount_in: U256,
    pub max_splits: usize,
}

#[derive(Clone, Debug)]
pub struct SwapAllocation {
    pub pool: Address,
    pub amount_in: U256,
}

#[derive(Clone, Debug)]
pub struct ExactInSwapResult {
    pub total_out: U256,
    pub allocations: Vec<SwapAllocation>, // guaranteed ≤ max_splits
}

#[derive(Debug)]
pub enum RouteError {
    /// None of the supplied pools match the desired token order.
    NoViablePool,
    /// amount_in == 0
    AmountTooSmall,
    /// max_splits == 0
    SplitsLimitTooLow,
    /// The math failed (e.g., EulerSwap helper returned Err).
    ComputationFailed(CurveError),
}

impl From<CurveError> for RouteError {
    fn from(error: CurveError) -> Self {
        RouteError::ComputationFailed(error)
    }
}

impl From<MathError> for RouteError {
    fn from(error: MathError) -> Self {
        RouteError::ComputationFailed(CurveError::Math(error))
    }
}
