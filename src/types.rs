// SPDX-License-Identifier: Apache-2.0

// types.rs for hwx
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HwxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("Invalid PTX code: {0}")]
    InvalidPtx(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, HwxError>;

#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult {
    pub hwx_type: HwxType,
    pub element_count: usize,
    pub numeric_value_1: f64,
    pub numeric_value_2: f64,
}

impl ClassificationResult {
    #[inline]
    pub fn new(hwx_type: HwxType) -> Self {
        Self {
            hwx_type,
            element_count: 1,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        }
    }

    #[inline]
    pub fn new_with_count(hwx_type: HwxType, element_count: usize) -> Self {
        Self {
            hwx_type,
            element_count,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    pub time: u64,
    pub value: f64,
}

impl MetricPoint {
    pub fn new(time: u64, value: f64) -> Self {
        Self { time, value }
    }

    pub fn get_time(&self) -> u64 {
        self.time
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }
}

/// A lightweight "event" container used by HWX's field/string matching helpers.
///
/// HWX does not assume any particular upstream event/log schema. If you already have
/// your own event type, you can map it into this representation (or wrap/convert it)
/// before calling the helper functions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StoredEvent {
    fields: Vec<StoredField>,
}

impl StoredEvent {
    #[inline]
    pub fn new(fields: Vec<StoredField>) -> Self {
        Self { fields }
    }

    #[inline]
    pub fn get_fields(&self) -> &[StoredField] {
        &self.fields
    }

    /// Returns a concatenated view of all field values, separated by spaces.
    ///
    /// This is a convenience helper for "search across all fields" behavior.
    pub fn get_text(&self) -> String {
        if self.fields.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        for (i, f) in self.fields.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(f.get_value());
        }
        out
    }

    /// Returns the non-numeric fields for compatibility with upstream schemas.
    ///
    /// HWX does not enforce numeric/non-numeric separation; by default all fields
    /// are treated as non-numeric.
    #[inline]
    pub fn get_non_numeric_fields(&self) -> Option<&[StoredField]> {
        Some(&self.fields)
    }

    /// Returns the numeric fields for compatibility with upstream schemas.
    ///
    /// HWX does not enforce numeric/non-numeric separation; by default this is `None`.
    #[inline]
    pub fn get_numeric_fields(&self) -> Option<&[StoredField]> {
        None
    }

    #[inline]
    pub fn push_field(&mut self, field: StoredField) {
        self.fields.push(field);
    }
}

/// A simple key/value field used by [`StoredEvent`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredField {
    name: String,
    value: String,
}

impl StoredField {
    #[inline]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }

    #[inline]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn get_value(&self) -> &str {
        &self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum HwxType {
    DateMath,
    LogDate,
    ISO8601Date,
    FullDate,
    RFC2822Date,
    AmericanDate,
    EuropeanDate,
    VerboseDate,
    FinancialDate,
    DMYDate,
    GenericDate,
    IPAddressV4,
    IPAddressV6,
    Geo,
    Integer,
    Float,
    File,
    String,
    Boolean,
    Null,
    Vector,
    Array,
    IntegerArray,
    FloatArray,
    StringArray,
    BooleanArray,
    DateMathArray,
    LogDateArray,
    ISO8601DateArray,
    FullDateArray,
    RFC2822DateArray,
    AmericanDateArray,
    EuropeanDateArray,
    VerboseDateArray,
    FinancialDateArray,
    DMYDateArray,
    GenericDateArray,
    IPAddressV4Array,
    IPAddressV6Array,
    GeoArray,
    FileArray,
    Object,
    Nested,
    Undefined,
}
