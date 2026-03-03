//! Progress-bar helper for benchmark runs.
//!
//! The benchmarks are often long-running; this module encapsulates the formatting and
//! visibility toggle for the `indicatif` progress bar so callers can opt in/out with a
//! simple boolean.

use indicatif::{ProgressBar, ProgressStyle};

/// Create a configured progress bar, returning `None` when `visible` is `false`.
pub fn make_bar(len: u64, visible: bool) -> Option<ProgressBar> {
    if !visible {
        return None;
    }
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner} {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise} | eta {eta_precise} | {msg}"
        ).unwrap().progress_chars("##-")
    );
    Some(pb)
}
