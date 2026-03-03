//! Timing helpers used by the benchmark harness.
//!
//! `time_millis` wraps a closure with optional warm-up, memory fences, and `black_box`
//! calls so we measure real work rather than compiler-optimised artefacts.

use std::hint::black_box;
use std::sync::atomic::{fence, Ordering};
use std::time::Instant;

/// Runs `f` after an optional warm-up, with memory fences and black_box to
/// reduce cross-optimization effects. Returns elapsed millis.
pub fn time_millis<F, R>(mut f: F, warmup: bool) -> (u128, Option<R>)
where
    F: FnMut() -> R,
{
    if warmup {
        let _ = black_box(f()); // warm path & caches
    }
    fence(Ordering::SeqCst);
    let t0 = Instant::now();
    let out = black_box(f());
    fence(Ordering::SeqCst);
    (t0.elapsed().as_millis(), Some(out))
}

/// Runs `f` after an optional warm-up, with memory fences and black_box to
/// reduce cross-optimization effects. Returns elapsed nanos.
///
/// Note: `Instant` resolution is platform dependent; ns here is the Duration unit,
/// not a guarantee of true hardware-nanosecond precision.
pub fn time_nanos<F, R>(mut f: F, warmup: bool) -> (u128, Option<R>)
where
    F: FnMut() -> R,
{
    if warmup {
        let _ = black_box(f()); // warm path & caches
    }
    fence(Ordering::SeqCst);
    let t0 = Instant::now();
    let out = black_box(f());
    fence(Ordering::SeqCst);
    (t0.elapsed().as_nanos(), Some(out))
}
