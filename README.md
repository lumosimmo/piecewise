# Piecewise

**Piecewise** is a high-performance, open-source Rust library for aggregating trades across multiple EulerSwap pools. It uses a greedy marginal-price equalization algorithm to distribute a trade across several pools, minimizing slippage and maximizing the total output amount.

This library is designed to be used as a core component in applications requiring efficient trade execution on EulerSwap, such as trading bots, and generic aggregators.

## Motivation

EulerSwap only support 1 liquidity provider per pool, which means that for popular token pairs, the liquidity is fragmented across many pools. This makes it difficult to execute large trades efficiently.

Piecewise helps solve this problem by aggregating trades across multiple pools sharing the same token pair. With the output of the aggregator, traders can execute optimized swaps via a router as if they were using a single pool.

## Core Functionality

The heart of the aggregator is the `find_best_route_exact_in` function. This function takes a set of available liquidity pools and a swap request, and returns a detailed execution plan.

The algorithm works as follows:

1.  **Sorting**: Pools are sorted from the best price to the worst.
2.  **Greedy Fill**: The algorithm iteratively allocates the input amount to the pool with the current best marginal price.
3.  **Price Equalization**: It calculates the exact input amount required to shift the best pool's price down to match the price of the second-best pool.
4.  **Iteration**: This process repeats, treating the now-equalized top pools as a single unit, until the entire input amount is allocated, or the maximum number of splits specified by the user is reached.

This method ensures that the trade is spread across the most price-efficient pools, resulting in a better overall execution price than using a single pool.

## Crate Structure

The library is organized into two main modules:

- `math`: Contains all the core mathematical utilities for dealing with EulerSwap's bonding curve formula, including functions for price calculation, reserve state computation, and quoting. Not all functions are used in the solver, but I added them for completeness.
- `solver`: Implements the trade routing algorithm itself.

## Future Work

Currently, the library only supports exact-in swaps. Exact-out swaps can be supported with a bit more work (perhaps with binary search).

The price precision used internally is 18-decimal fixed point, I have added 27-decimal helpers but haven't fully tested them yet. Usually, this precision is enough for any major token pair. Problems can arise when the USD price difference between the token pair is too large - for example, a 18-decimal meme coin paired with 6-decimal USDC. This will be addressed soon.

For simplicity, the library also does not consider how much "un-utilized" liquidity is left in the pools, which can yield invalid routes for large trades when an Euler position is already tilted to one side and doesn't have enough liquidity to cover the entire trade. This will be addressed in the future with additional pool state information like the [limits](https://github.com/euler-xyz/euler-swap/blob/a988dc551c223f84ab78cfd519e3e70082488624/src/libraries/QuoteLib.sol#L42) returned by the contract.

The main purpose of this library is to provide a simple solution to the EulerSwap liquidity fragmentation problem, therefore in the current state, it doesn't support routing through pools with different token pairs, and thus it's not a generic aggregator like 1inch.
