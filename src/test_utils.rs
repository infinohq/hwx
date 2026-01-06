// SPDX-License-Identifier: Apache-2.0

/// Test-only helpers.
///
/// Keep this module lightweight and dependency-free so `cargo test` works out of the box.
pub fn config_test_logger() {
    // Intentionally a no-op.
    // Some tests call this to enable logging in downstream repos; HWX doesn't require
    // a logger for correctness.
}
