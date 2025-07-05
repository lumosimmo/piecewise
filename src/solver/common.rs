use alloy_primitives::{Address, U256};
use serde::{Deserialize, Serialize};

use crate::math::common::MathError;
use crate::math::curve::{CurveError, EulerSwapParams};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolSnapshot {
    pub pool_address: Address,
    pub token0: Address,
    pub token1: Address,
    pub reserve0: U256,
    pub reserve1: U256,
    pub params: EulerSwapParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactInSwapRequest {
    pub token_in: Address,
    pub token_out: Address,
    pub amount_in: U256,
    #[cfg_attr(
        target_arch = "wasm32",
        serde(serialize_with = "crate::wasm_serde::serialize_usize")
    )]
    pub max_splits: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwapAllocation {
    pub pool: Address,
    pub amount_in: U256,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactInSwapResult {
    pub total_out: U256,
    pub allocations: Vec<SwapAllocation>, // guaranteed ≤ max_splits
}

#[derive(Debug, Serialize, Deserialize)]
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

#[cfg(target_arch = "wasm32")]
impl From<RouteError> for serde_wasm_bindgen::Error {
    fn from(err: RouteError) -> Self {
        serde_wasm_bindgen::Error::from(
            serde_wasm_bindgen::to_value(&err)
                .unwrap_or_else(|e| wasm_bindgen::JsValue::from_str(&e.to_string())),
        )
    }
}
