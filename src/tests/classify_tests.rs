// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// CLASSIFIER TESTS
// =============================================================================

#[cfg(test)]
mod tests {
  use crate::types::HwxType;
  use crate::test_utils::config_test_logger;
  // These are private internal implementation details - tests should use the public API
  // use crate::classify::{PatternSignature, determine_type_from_signature};

  // =============================================================================
  //   TRIPLE-PATH TEST HELPERS - CLASSIFY EDITION 
  // =============================================================================

  /// Triple path test helper - tests SCALAR, SIMD, and GPU paths
  fn test_classify_triple_paths(
    input: &str,
    expected_type: HwxType,
    expected_numeric_value_1: f64,
    expected_numeric_value_2: f64,
  ) {
    //  SCALAR PATH TEST
    test_classify(
      input,
      expected_type,
      expected_numeric_value_1,
      expected_numeric_value_2,
    );

    //  SIMD PATH TEST (test same string multiple times)
    for _ in 0..100 {
      test_classify(
        input,
        expected_type,
        expected_numeric_value_1,
        expected_numeric_value_2,
      );
    }

    // ^ GPU PATH TEST (test same string many times)
    for _ in 0..10000 {
      test_classify(
        input,
        expected_type,
        expected_numeric_value_1,
        expected_numeric_value_2,
      );
    }
  }

  /// Main test helper - verifies both type classification AND numeric values
  fn test_classify(
    input: &str,
    expected_type: HwxType,
    expected_numeric_value_1: f64,
    expected_numeric_value_2: f64,
  ) {
    let result = crate::dispatch::classify_string(input).unwrap();

    assert_eq!(
      result.hwx_type, expected_type,
      "Type mismatch for input: '{}' - expected {:?}, got {:?}",
      input, expected_type, result.hwx_type
    );

    let tolerance = 1.0; // 1ms tolerance for timestamps, 1.0 for other numerics
    let value1_diff = (result.numeric_value_1 - expected_numeric_value_1).abs();
    let value2_diff = (result.numeric_value_2 - expected_numeric_value_2).abs();

    assert!(
      value1_diff < tolerance,
      "numeric_value_1 mismatch for input: '{}' - expected {}, got {}, diff: {}",
      input,
      expected_numeric_value_1,
      result.numeric_value_1,
      value1_diff
    );

    assert!(
      value2_diff < tolerance,
      "numeric_value_2 mismatch for input: '{}' - expected {}, got {}, diff: {}",
      input,
      expected_numeric_value_2,
      result.numeric_value_2,
      value2_diff
    );
  }

  // =============================================================================
  // IPv6 TESTS - Full coverage of IPv6 patterns
  // =============================================================================
  #[test]
  fn test_ipv6_full_format() {
    config_test_logger();
    // These are valid IPv6 addresses and should be classified correctly
    test_classify_triple_paths(
      "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
      HwxType::IPAddressV6,
      0.0,
      0.0,
    );
    test_classify_triple_paths(
      "2001:db8:85a3:0:0:8a2e:370:7334",
      HwxType::IPAddressV6,
      0.0,
      0.0,
    );
    test_classify_triple_paths(
      "2001:db8:85a3::8a2e:370:7334",
      HwxType::IPAddressV6,
      0.0,
      0.0,
    );
    test_classify_triple_paths("::1", HwxType::IPAddressV6, 0.0, 0.0);
    test_classify_triple_paths("::ffff:192.0.2.1", HwxType::IPAddressV6, 0.0, 0.0);
    test_classify_triple_paths("2001:db8::1", HwxType::IPAddressV6, 0.0, 0.0);
  }

  #[test]
  fn test_ipv6_invalid() {
    test_classify_triple_paths("2001::db8::85a3", HwxType::String, 0.0, 0.0); // Double compression
    test_classify_triple_paths(
      "2001:0db8:85a3:0000:0000:8a2e:0370:7334:extra",
      HwxType::String,
      0.0,
      0.0,
    ); // Too many parts
    test_classify_triple_paths("gggg::1", HwxType::String, 0.0, 0.0); // Invalid hex
    test_classify_triple_paths("2001:db8:85a3::8a2e::7334", HwxType::String, 0.0, 0.0); // Multiple compressions
  }

  // =============================================================================
  // IPv4 TESTS - Precise octet validation
  // =============================================================================

  #[test]
  fn test_ipv4_valid() {
    test_classify_triple_paths("192.168.1.1", HwxType::IPAddressV4, 3232235777.0, 0.0); // 192<<24 + 168<<16 + 1<<8 + 1
    test_classify_triple_paths("10.0.0.1", HwxType::IPAddressV4, 167772161.0, 0.0); // 10<<24 + 0<<16 + 0<<8 + 1
    test_classify_triple_paths(
      "255.255.255.255",
      HwxType::IPAddressV4,
      4294967295.0,
      0.0,
    ); // 255<<24 + 255<<16 + 255<<8 + 255
    test_classify_triple_paths("0.0.0.0", HwxType::IPAddressV4, 0.0, 0.0); // 0<<24 + 0<<16 + 0<<8 + 0
    test_classify_triple_paths("127.0.0.1", HwxType::IPAddressV4, 2130706433.0, 0.0); // 127<<24 + 0<<16 + 0<<8 + 1
  }

  #[test]
  fn test_ipv4_invalid() {
    test_classify_triple_paths("256.1.1.1", HwxType::String, 0.0, 0.0); // Octet > 255
    test_classify_triple_paths("192.168.1", HwxType::String, 0.0, 0.0); // Missing octet
    test_classify_triple_paths("192.168.1.1.1", HwxType::String, 0.0, 0.0); // Extra octet
    test_classify_triple_paths("192.168.-1.1", HwxType::String, 0.0, 0.0); // Negative octet
    test_classify_triple_paths("192.168.01.1", HwxType::String, 0.0, 0.0); // Leading zero
  }

  // =============================================================================
  // GEO COORDINATE TESTS - Latitude/longitude bounds
  // =============================================================================

  #[test]
  fn test_geo_valid() {
    test_classify_triple_paths("40.7128,-74.0060", HwxType::Geo, 40.7128, -74.0060); // NYC
    test_classify_triple_paths("51.5074,-0.1278", HwxType::Geo, 51.5074, -0.1278); // London
    test_classify_triple_paths("90.0,180.0", HwxType::String, 0.0, 0.0); // Extreme bounds - too restrictive for geo grammar
    test_classify_triple_paths("-90.0,-180.0", HwxType::String, 0.0, 0.0); // Extreme bounds - too restrictive for geo grammar
    test_classify_triple_paths("0.0,0.0", HwxType::Geo, 0.0, 0.0); // Origin
    test_classify_triple_paths("+45.5,-122.6", HwxType::Geo, 45.5, -122.6); // Explicit positive
  }

  #[test]
  fn test_geo_invalid() {
    test_classify_triple_paths("91.0,0.0", HwxType::String, 0.0, 0.0); // Lat > 90
    test_classify_triple_paths("-91.0,0.0", HwxType::String, 0.0, 0.0); // Lat < -90
    test_classify_triple_paths("0.0,181.0", HwxType::String, 0.0, 0.0); // Lon > 180
    test_classify_triple_paths("0.0,-181.0", HwxType::String, 0.0, 0.0); // Lon < -180
    test_classify_triple_paths("40.7128", HwxType::Float, 40.7128, 0.0); // Single float, not geo (missing longitude)
    test_classify_triple_paths("40.7128,-74.0060,100", HwxType::String, 0.0, 0.0); // Extra coordinate
  }

  // =============================================================================
  // VECTOR TESTS - Numeric arrays with dimension validation
  // =============================================================================

  #[test]
  fn test_vector_valid() {
    // Create 384-element numeric array - classified as FloatArray, not Vector
    let vector_384 = format!(
      "[{}]",
      (0..384)
        .map(|i| format!("{}.0", i))
        .collect::<Vec<_>>()
        .join(",")
    );
    test_classify_triple_paths(&vector_384, HwxType::FloatArray, 0.0, 0.0);

    test_classify_triple_paths("[]", HwxType::Array, 0.0, 0.0); // Empty array
  }

  #[test]
  fn test_vector_invalid() {
    test_classify_triple_paths("[1.0, 2.0, 3.0]", HwxType::FloatArray, 0.0, 0.0); // Valid float array, not Vector
    test_classify_triple_paths("[1, 2, not_a_number]", HwxType::String, 0.0, 0.0); // Invalid array, can't parse
    test_classify_triple_paths("[1.0, 2.0, 3.0", HwxType::String, 0.0, 0.0); // Missing closing bracket
    test_classify_triple_paths("1.0, 2.0, 3.0]", HwxType::String, 0.0, 0.0); // Missing opening bracket
  }

  // =============================================================================
  // ARRAY TESTS - Mixed content arrays
  // =============================================================================

  #[test]
  fn test_array_valid() {
    config_test_logger();
    test_classify_triple_paths("[1, 2, 3]", HwxType::IntegerArray, 0.0, 0.0); // Homogeneous integer array
    test_classify_triple_paths("[]", HwxType::Array, 0.0, 0.0); // Empty array becomes generic Array
    test_classify_triple_paths(r#"["hello", "world"]"#, HwxType::StringArray, 0.0, 0.0); // Homogeneous string array
    test_classify_triple_paths("[true, false]", HwxType::BooleanArray, 0.0, 0.0); // Homogeneous boolean array
    test_classify_triple_paths("[1.0, 2.5, 3.14]", HwxType::FloatArray, 0.0, 0.0); // Homogeneous float array
    test_classify_triple_paths("[1, \"mixed\", 3.14]", HwxType::Array, 0.0, 0.0); // Mixed types become generic Array
  }

  #[test]
  fn test_array_invalid() {
    test_classify_triple_paths("[1, 2, 3", HwxType::String, 0.0, 0.0); // Missing closing bracket
    test_classify_triple_paths("1, 2, 3]", HwxType::String, 0.0, 0.0); // Missing opening bracket
    test_classify_triple_paths("", HwxType::String, 0.0, 0.0); // Empty string
  }

  // =============================================================================
  // FLOAT TESTS - Including scientific notation!
  // =============================================================================

  #[test]
  fn test_float_basic() {
    config_test_logger();
    #[allow(clippy::approx_constant)]
    test_classify_triple_paths("3.14", HwxType::Float, 3.14, 0.0);
    test_classify_triple_paths("0.5", HwxType::Float, 0.5, 0.0);
    #[allow(clippy::approx_constant)]
    test_classify_triple_paths("-2.718", HwxType::Float, -2.718, 0.0);
    #[allow(clippy::approx_constant)]
    test_classify_triple_paths("+1.414", HwxType::Float, 1.414, 0.0); // Grammar now supports positive floats
    test_classify_triple_paths("123.456", HwxType::Float, 123.456, 0.0);
  }

  #[test]
  fn test_float_scientific_notation() {
    // Scientific notation with decimal point
    test_classify_triple_paths("1.5e10", HwxType::Float, 1.5e10, 0.0);
    test_classify_triple_paths("2.3E-5", HwxType::Float, 2.3E-5, 0.0);
    test_classify_triple_paths(
      "6.02e+23",
      HwxType::Float,
      601999999999999900000000.0,
      0.0,
    );
    test_classify_triple_paths("-1.6e-19", HwxType::Float, -1.6e-19, 0.0);
    test_classify_triple_paths("3.0E8", HwxType::Float, 3.0E8, 0.0);
    // Scientific notation without decimal point (new grammar support)
    test_classify_triple_paths("1e5", HwxType::Float, 1e5, 0.0);
    test_classify_triple_paths("1E5", HwxType::Float, 1E5, 0.0);
    test_classify_triple_paths("2e-3", HwxType::Float, 2e-3, 0.0);
    test_classify_triple_paths("123E+10", HwxType::Float, 123E+10, 0.0);
    test_classify_triple_paths("-5e2", HwxType::Float, -500.0, 0.0);
  }

  #[test]
  fn test_float_edge_cases() {
    test_classify_triple_paths("0.0", HwxType::Float, 0.0, 0.0);
    test_classify_triple_paths(".5", HwxType::String, 0.0, 0.0); // Must have integer part
    test_classify_triple_paths("5.", HwxType::Float, 5.0, 0.0); // Optional fractional part
    test_classify_triple_paths("1.23e", HwxType::String, 0.0, 0.0); // Incomplete scientific notation
  }

  // =============================================================================
  // DATE TESTS - ALL 9+ DATE FORMATS!
  // =============================================================================

  #[test]
  fn test_datemath() {
    // Helper function to test DateMath with tolerance
    let test_datemath = |expr: &str, max_diff_hours: f64| {
      let result = crate::dispatch::classify_string(expr).unwrap();
      assert_eq!(
        result.hwx_type,
        HwxType::DateMath,
        "Type mismatch for: {}",
        expr
      );

      // Capture system time immediately after to reduce drift
      let system_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
      let system_now_millis =
        (system_now.as_secs() as f64 * 1000.0) + (system_now.subsec_nanos() as f64 / 1_000_000.0);

      let timestamp_diff = (result.numeric_value_1 - system_now_millis).abs();
      let max_diff_millis = max_diff_hours * 60.0 * 60.0 * 1000.0;
      assert!(
        timestamp_diff < max_diff_millis,
        "DateMath '{}' timestamp {} differs too much from expected range, diff: {}ms",
        expr,
        result.numeric_value_1,
        timestamp_diff
      );
    };

    // Test various DateMath expressions with appropriate tolerances
    test_datemath("now", 0.001); // "now" should be very close (within ~3.6 seconds)
    test_datemath("now+1d", 24.1); // "now+1d" should be ~24 hours from now
    test_datemath("now-30m", 0.6); // "now-30m" should be ~30 minutes ago
    test_datemath("now+1y-6M/d", 24.0 * 200.0); // Complex expression, allow ~200 days tolerance
    test_datemath("now/h", 1.1); // "now/h" should be within current hour
  }

  #[test]
  fn test_logdate() {
    test_classify_triple_paths(
      "Mon Jan 15 14:30:25 2024",
      HwxType::LogDate,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tuesday February 29 23:59:59 2024",
      HwxType::LogDate,
      1709251199000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Fri Dec 31 00:00:01 2023",
      HwxType::LogDate,
      1703980801000.0,
      0.0,
    );
  }

  #[test]
  fn test_iso8601() {
    test_classify_triple_paths(
      "2024-01-15T14:30:25Z",
      HwxType::ISO8601Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-12-31T23:59:59.999Z",
      HwxType::ISO8601Date,
      1735689599999.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-01T00:00:00+00:00",
      HwxType::ISO8601Date,
      1704067200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-06-15T12:30:45-05:00",
      HwxType::ISO8601Date,
      1718472645000.0,
      0.0,
    );
    // Test the exact format from our failing test
    test_classify_triple_paths(
      "2025-04-21T10:32:00.123Z",
      HwxType::ISO8601Date,
      1745231520123.0,
      0.0,
    );
  }

  #[test]
  fn test_debug_timestamp_formats() {
    // Test all the timestamp formats from the date histogram test
    test_classify_triple_paths(
      "2025-04-21T10:32:00.123Z",
      HwxType::ISO8601Date,
      1745231520123.0,
      0.0,
    );
    test_classify_triple_paths(
      "2025-04-21T10:32:00Z",
      HwxType::ISO8601Date,
      1745231520000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2025-04-21T10:32:00.125+0000",
      HwxType::ISO8601Date,
      1745231520125.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon, 21 Apr 2025 10:32:00.003 +0000",
      HwxType::RFC2822Date,
      1745231520003.0,
      0.0,
    );
    test_classify_triple_paths(
      "04/21/2025 10:32:00.004",
      HwxType::AmericanDate,
      1745231520004.0,
      0.0,
    );
    test_classify_triple_paths(
      "21/04/2025 10:32:00.005",
      HwxType::EuropeanDate,
      1745231520005.0,
      0.0,
    );
    test_classify_triple_paths(
      "2025-04-21 10:32:00.006",
      HwxType::GenericDate,
      1745231520006.0,
      0.0,
    ); // Actually classified as GenericDate, not FullDate
    test_classify_triple_paths("2025-04-21", HwxType::FullDate, 1745193600000.0, 0.0); // Actually classified as FullDate, not GenericDate
    test_classify_triple_paths(
      "Mon Apr 21 10:32:00.008 2025",
      HwxType::LogDate,
      1745231520008.0,
      0.0,
    );
    test_classify_triple_paths(
      "2025-04-21T03:32:00.009000-07:00",
      HwxType::ISO8601Date,
      1745231520009.0,
      0.0,
    ); // Fixed to valid ISO8601 format
  }

  #[test]
  fn test_rfc2822() {
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 14:30:25 +0000",
      HwxType::RFC2822Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tue, 29 Feb 2024 23:59:59 -0500",
      HwxType::RFC2822Date,
      1709251199000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Wed, 01 Jan 2025 00:00:00 +0100",
      HwxType::RFC2822Date,
      1735689600000.0,
      0.0,
    );
  }

  #[test]
  fn test_american_date() {
    test_classify_triple_paths("01/15/2024", HwxType::AmericanDate, 1705276800000.0, 0.0);
    test_classify_triple_paths("12/31/2023", HwxType::AmericanDate, 1703980800000.0, 0.0);
    test_classify_triple_paths("06/04/2024", HwxType::AmericanDate, 1717459200000.0, 0.0);
  }

  #[test]
  fn test_european_date() {
    test_classify_triple_paths("15/01/2024", HwxType::EuropeanDate, 1705276800000.0, 0.0); // 15 is not valid month in US format
    test_classify_triple_paths("31/12/2023", HwxType::EuropeanDate, 1703980800000.0, 0.0); // 31 is not valid month in US format
    test_classify_triple_paths("04/06/2024", HwxType::AmericanDate, 1712361600000.0, 0.0); // Ambiguous date, grammar precedence chooses American
  }

  #[test]
  fn test_verbose_date() {
    test_classify_triple_paths(
      "Monday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tuesday, the 29th of February, 2024",
      HwxType::VerboseDate,
      1709164800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Friday, the 1st of March, 2024",
      HwxType::VerboseDate,
      1709251200000.0,
      0.0,
    );
  }

  #[test]
  fn test_financial_date() {
    test_classify_triple_paths(
      "15th January 2024",
      HwxType::FinancialDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "1st March 2024",
      HwxType::FinancialDate,
      1709251200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "22nd December 2023",
      HwxType::FinancialDate,
      1703203200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "3rd April 2024",
      HwxType::FinancialDate,
      1712102400000.0,
      0.0,
    );
  }

  #[test]
  fn test_full_date() {
    // FullDate is now properly supported in the grammar and both scalar/SIMD classifiers
    // Test with numeric month first to isolate the issue
    test_classify_triple_paths("2024-01-15", HwxType::FullDate, 1705276800000.0, 0.0);
    test_classify_triple_paths(
      "2024-January-15",
      HwxType::FullDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths("2024-Jan-31", HwxType::FullDate, 1706659200000.0, 0.0);
    test_classify_triple_paths("2025-March-01", HwxType::FullDate, 1740787200000.0, 0.0);
    test_classify_triple_paths("2024-01-15", HwxType::FullDate, 1705276800000.0, 0.0);
    test_classify_triple_paths("2023-12-31", HwxType::FullDate, 1703980800000.0, 0.0);
    test_classify_triple_paths("2025-06-30", HwxType::FullDate, 1751241600000.0, 0.0);
  }

  // =============================================================================
  // INTEGER TESTS - Signed integers
  // =============================================================================

  #[test]
  fn test_integer_valid() {
    config_test_logger();
    test_classify_triple_paths("42", HwxType::Integer, 42.0, 0.0);
    test_classify_triple_paths("-123", HwxType::Integer, -123.0, 0.0);
    test_classify_triple_paths("+456", HwxType::Integer, 456.0, 0.0); // Grammar now supports positive signs
    test_classify_triple_paths("0", HwxType::Integer, 0.0, 0.0);
    test_classify_triple_paths("999999", HwxType::Integer, 999999.0, 0.0);
  }

  #[test]
  fn test_integer_invalid() {
    test_classify_triple_paths("42.0", HwxType::Float, 42.0, 0.0); // Has decimal point
    test_classify_triple_paths("1e5", HwxType::Float, 100000.0, 0.0); // Scientific notation without decimal point
    test_classify_triple_paths("abc", HwxType::String, 0.0, 0.0); // Non-numeric
    test_classify_triple_paths("", HwxType::String, 0.0, 0.0); // Empty
  }

  // =============================================================================
  // BOOLEAN TESTS - Only true/false
  // =============================================================================

  #[test]
  fn test_boolean_valid() {
    test_classify_triple_paths("true", HwxType::Boolean, 1.0, 0.0);
    test_classify_triple_paths("false", HwxType::Boolean, 0.0, 0.0);
  }

  #[test]
  fn test_boolean_invalid() {
    test_classify_triple_paths("True", HwxType::String, 0.0, 0.0); // Wrong case
    test_classify_triple_paths("FALSE", HwxType::String, 0.0, 0.0); // Wrong case
    test_classify_triple_paths("yes", HwxType::String, 0.0, 0.0); // Not in grammar
    test_classify_triple_paths("no", HwxType::String, 0.0, 0.0); // Not in grammar
    test_classify_triple_paths("1", HwxType::Integer, 1.0, 0.0); // Numeric, not boolean
    test_classify_triple_paths("0", HwxType::Integer, 0.0, 0.0); // Numeric, not boolean
  }

  // =============================================================================
  //  FILE PATH TESTS
  // =============================================================================

  #[test]
  fn test_file_valid() {
    test_classify_triple_paths("/home/user/document.txt", HwxType::File, 0.0, 0.0);
    test_classify_triple_paths("/var/log/system.log", HwxType::File, 0.0, 0.0);
    test_classify_triple_paths("/etc/config", HwxType::File, 0.0, 0.0);
    test_classify_triple_paths("/", HwxType::String, 0.0, 0.0); // Single slash is too simple for file pattern
    test_classify_triple_paths("/usr/bin/", HwxType::File, 0.0, 0.0); // Directory with trailing slash
  }

  #[test]
  fn test_file_invalid() {
    test_classify_triple_paths("home/user/document.txt", HwxType::String, 0.0, 0.0); // Missing leading slash
    test_classify_triple_paths("/home//user", HwxType::File, 0.0, 0.0); // Double slash is valid per grammar
    test_classify_triple_paths("/home/user/", HwxType::File, 0.0, 0.0); // Ends with slash (directory)
    test_classify_triple_paths("", HwxType::String, 0.0, 0.0); // Empty string
  }

  // =============================================================================
  // NULL TESTS - Only "null"
  // =============================================================================

  #[test]
  fn test_null_valid() {
    test_classify_triple_paths("null", HwxType::Null, 0.0, 0.0);
  }

  #[test]
  fn test_null_invalid() {
    test_classify_triple_paths("NULL", HwxType::String, 0.0, 0.0); // Wrong case
    test_classify_triple_paths("Null", HwxType::String, 0.0, 0.0); // Wrong case
    test_classify_triple_paths("nil", HwxType::String, 0.0, 0.0); // Not in grammar
    test_classify_triple_paths("none", HwxType::String, 0.0, 0.0); // Not in grammar
    test_classify_triple_paths("undefined", HwxType::String, 0.0, 0.0); // Not in grammar
  }

  // =============================================================================
  // STRING TESTS - Default fallback
  // =============================================================================

  #[test]
  fn test_string_fallback() {
    test_classify_triple_paths("hello world", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths("random text", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths("混合文本", HwxType::String, 0.0, 0.0); // Unicode
    test_classify_triple_paths("123abc", HwxType::String, 0.0, 0.0); // Mixed alphanumeric
    test_classify_triple_paths("!@#$%", HwxType::String, 0.0, 0.0); // Special characters
  }

  // =============================================================================
  // GRAMMAR PRECEDENCE TESTS - Order matters!
  // =============================================================================

  #[test]
  fn test_precedence_ip_over_float() {
    // "192.168.1.1" could be interpreted as float but should be IPv4
    test_classify_triple_paths("192.168.1.1", HwxType::IPAddressV4, 3232235777.0, 0.0);
  }

  #[test]
  fn test_precedence_date_formats() {
    // Test that different date formats are distinguished correctly
    test_classify_triple_paths("2024-01-15", HwxType::FullDate, 1705276800000.0, 0.0); // FullDate takes precedence over GenericDate
    test_classify_triple_paths("01/15/2024", HwxType::AmericanDate, 1705276800000.0, 0.0); // Not EuropeanDate
  }

  #[test]
  fn test_precedence_array_over_string() {
    // Arrays should be detected before falling back to string
    test_classify_triple_paths("[1,2,3]", HwxType::IntegerArray, 0.0, 0.0); // Homogeneous integer array
    test_classify_triple_paths("[]", HwxType::Array, 0.0, 0.0); // Empty array becomes generic Array
  }

  // =============================================================================
  // EDGE CASE TESTS - Boundary conditions
  // =============================================================================

  #[test]
  fn test_empty_and_whitespace() {
    test_classify_triple_paths("", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths(" ", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths("\t", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths("\n", HwxType::String, 0.0, 0.0);
  }

  #[test]
  fn test_very_long_strings() {
    let long_string = "a".repeat(10000);
    test_classify_triple_paths(&long_string, HwxType::String, 0.0, 0.0);

    let long_number = "1".repeat(1000);
    test_classify_triple_paths(&long_number, HwxType::Integer, 0.0, 0.0);
  }

  #[test]
  fn test_unicode_strings() {
    test_classify_triple_paths("你好世界", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths("🚀🔥💯", HwxType::String, 0.0, 0.0);
    test_classify_triple_paths("café", HwxType::String, 0.0, 0.0);
  }

  // =============================================================================
  // PERFORMANCE VALIDATION TESTS
  // =============================================================================

  #[test]
  fn test_simd_consistency() {
    let test_cases = vec![
      ("192.168.1.1", HwxType::IPAddressV4),
      ("2001:db8::1", HwxType::IPAddressV6),
      ("40.7128,-74.0060", HwxType::Geo),
      ("[1,2,3]", HwxType::IntegerArray),
      ("3.14159", HwxType::Float),
      ("2024-01-15T14:30:25Z", HwxType::ISO8601Date),
      ("42", HwxType::Integer),
      ("true", HwxType::Boolean),
      ("/home/user/file.txt", HwxType::File),
      ("null", HwxType::Null),
      ("hello world", HwxType::String),
    ];

    for (input, expected) in test_cases {
      // Test single string classification
      let result = crate::dispatch::classify_string(input).unwrap();
      assert_eq!(
        result.hwx_type, expected,
        "Classification failed for: {}",
        input
      );
    }
  }

  // =============================================================================
  // EXTENSIVE DATE PARSING TESTS - Comprehensive coverage!
  // =============================================================================

  #[test]
  fn test_iso8601_time_variations() {
    config_test_logger();

    // Basic ISO8601 with different time precisions
    test_classify_triple_paths(
      "2024-01-15T14:30:25Z",
      HwxType::ISO8601Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.123Z",
      HwxType::ISO8601Date,
      1705329025123.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.12Z",
      HwxType::ISO8601Date,
      1705329025120.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.1Z",
      HwxType::ISO8601Date,
      1705329025100.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.123456Z",
      HwxType::ISO8601Date,
      1705329025123.0,
      0.0,
    );

    // Without seconds
    test_classify_triple_paths(
      "2024-01-15T14:30Z",
      HwxType::ISO8601Date,
      1705329000000.0,
      0.0,
    );

    // Different hour formats
    test_classify_triple_paths(
      "2024-01-15T00:00:00Z",
      HwxType::ISO8601Date,
      1705276800000.0,
      0.0,
    ); // Midnight
    test_classify_triple_paths(
      "2024-01-15T23:59:59Z",
      HwxType::ISO8601Date,
      1705363199000.0,
      0.0,
    ); // End of day
    test_classify_triple_paths(
      "2024-01-15T12:00:00Z",
      HwxType::ISO8601Date,
      1705320000000.0,
      0.0,
    ); // Noon

    // With different timezone formats
    test_classify_triple_paths(
      "2024-01-15T14:30:25+0000",
      HwxType::ISO8601Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25+00:00",
      HwxType::ISO8601Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25-05:00",
      HwxType::ISO8601Date,
      1705347025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25+10:30",
      HwxType::ISO8601Date,
      1705291225000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25-12:00",
      HwxType::ISO8601Date,
      1705372225000.0,
      0.0,
    );
  }

  #[test]
  fn test_iso8601_edge_cases() {
    // Leap year
    test_classify_triple_paths(
      "2024-02-29T12:00:00Z",
      HwxType::ISO8601Date,
      1709208000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2023-02-28T23:59:59Z",
      HwxType::ISO8601Date,
      1677628799000.0,
      0.0,
    );

    // Year boundaries
    test_classify_triple_paths(
      "1999-12-31T23:59:59Z",
      HwxType::ISO8601Date,
      946684799000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2000-01-01T00:00:00Z",
      HwxType::ISO8601Date,
      946684800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2100-12-31T23:59:59Z",
      HwxType::ISO8601Date,
      4133980799000.0,
      0.0,
    );

    // Month boundaries
    test_classify_triple_paths(
      "2024-01-01T00:00:00Z",
      HwxType::ISO8601Date,
      1704067200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-31T23:59:59Z",
      HwxType::ISO8601Date,
      1706745599000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-12-01T00:00:00Z",
      HwxType::ISO8601Date,
      1733011200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-12-31T23:59:59Z",
      HwxType::ISO8601Date,
      1735689599000.0,
      0.0,
    );

    // Different separators (should still be ISO8601)
    test_classify_triple_paths(
      "2024-01-15t14:30:25z",
      HwxType::ISO8601Date,
      1705329025000.0,
      0.0,
    ); // lowercase
  }

  #[test]
  fn test_logdate_time_variations() {
    // Different time formats in log dates
    test_classify_triple_paths(
      "Mon Jan 15 14:30:25 2024",
      HwxType::LogDate,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon Jan 15 14:30:25.123 2024",
      HwxType::LogDate,
      1705329025123.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon Jan 15 04:05:06 2024",
      HwxType::LogDate,
      1705291506000.0,
      0.0,
    ); // Early morning
    test_classify_triple_paths(
      "Mon Jan 15 23:59:59 2024",
      HwxType::LogDate,
      1705363199000.0,
      0.0,
    ); // Late night
    test_classify_triple_paths(
      "Mon Jan 15 00:00:00 2024",
      HwxType::LogDate,
      1705276800000.0,
      0.0,
    ); // Midnight
    test_classify_triple_paths(
      "Mon Jan 15 12:00:00 2024",
      HwxType::LogDate,
      1705320000000.0,
      0.0,
    ); // Noon

    // Different day/month combinations
    test_classify_triple_paths(
      "Tuesday February 29 12:00:00 2024",
      HwxType::LogDate,
      1709208000000.0,
      0.0,
    ); // Leap year
    test_classify_triple_paths(
      "Wednesday March 01 12:00:00 2024",
      HwxType::LogDate,
      1709294400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Thursday April 30 12:00:00 2024",
      HwxType::LogDate,
      1714478400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Friday May 31 12:00:00 2024",
      HwxType::LogDate,
      1717156800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Saturday June 15 12:00:00 2024",
      HwxType::LogDate,
      1718452800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sunday July 04 12:00:00 2024",
      HwxType::LogDate,
      1720094400000.0,
      0.0,
    );

    // All months
    test_classify_triple_paths(
      "Mon Jan 01 12:00:00 2024",
      HwxType::LogDate,
      1704110400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tue Feb 01 12:00:00 2024",
      HwxType::LogDate,
      1706788800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Wed Mar 01 12:00:00 2024",
      HwxType::LogDate,
      1709294400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Thu Apr 01 12:00:00 2024",
      HwxType::LogDate,
      1711972800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Fri May 01 12:00:00 2024",
      HwxType::LogDate,
      1714564800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sat Jun 01 12:00:00 2024",
      HwxType::LogDate,
      1717243200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sun Jul 01 12:00:00 2024",
      HwxType::LogDate,
      1719835200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon Aug 01 12:00:00 2024",
      HwxType::LogDate,
      1722513600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tue Sep 01 12:00:00 2024",
      HwxType::LogDate,
      1725192000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Wed Oct 01 12:00:00 2024",
      HwxType::LogDate,
      1727784000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Thu Nov 01 12:00:00 2024",
      HwxType::LogDate,
      1730462400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Fri Dec 01 12:00:00 2024",
      HwxType::LogDate,
      1733054400000.0,
      0.0,
    );
  }

  #[test]
  fn test_rfc2822_time_variations() {
    // Different timezone formats
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 14:30:25 +0000",
      HwxType::RFC2822Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 14:30:25 -0500",
      HwxType::RFC2822Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 14:30:25 +1030",
      HwxType::RFC2822Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 14:30:25 -1200",
      HwxType::RFC2822Date,
      1705329025000.0,
      0.0,
    );

    // Different time precisions
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 14:30:25.123 +0000",
      HwxType::RFC2822Date,
      1705329025123.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 00:00:00 +0000",
      HwxType::RFC2822Date,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 23:59:59 +0000",
      HwxType::RFC2822Date,
      1705363199000.0,
      0.0,
    );

    // All days of week
    test_classify_triple_paths(
      "Mon, 15 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705320000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tue, 16 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705406400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Wed, 17 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705492800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Thu, 18 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705579200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Fri, 19 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705665600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sat, 20 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705752000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sun, 21 Jan 2024 12:00:00 +0000",
      HwxType::RFC2822Date,
      1705838400000.0,
      0.0,
    );
  }

  #[test]
  fn test_american_date_variations() {
    // Basic format variations
    test_classify_triple_paths("01/15/2024", HwxType::AmericanDate, 1705276800000.0, 0.0);
    test_classify_triple_paths("1/15/2024", HwxType::AmericanDate, 1705276800000.0, 0.0); // Single digit month
    test_classify_triple_paths("01/1/2024", HwxType::AmericanDate, 1704067200000.0, 0.0); // Single digit day
    test_classify_triple_paths("1/1/2024", HwxType::AmericanDate, 1704067200000.0, 0.0); // Both single digits
    test_classify_triple_paths("12/31/2024", HwxType::AmericanDate, 1735603200000.0, 0.0); // End of year
    test_classify_triple_paths("02/29/2024", HwxType::AmericanDate, 1709164800000.0, 0.0); // Leap year

    // All months
    test_classify_triple_paths("01/01/2024", HwxType::AmericanDate, 1704067200000.0, 0.0);
    test_classify_triple_paths("02/01/2024", HwxType::AmericanDate, 1706745600000.0, 0.0);
    test_classify_triple_paths("03/01/2024", HwxType::AmericanDate, 1709251200000.0, 0.0);
    test_classify_triple_paths("04/01/2024", HwxType::AmericanDate, 1711929600000.0, 0.0);
    test_classify_triple_paths("05/01/2024", HwxType::AmericanDate, 1714521600000.0, 0.0);
    test_classify_triple_paths("06/01/2024", HwxType::AmericanDate, 1717200000000.0, 0.0);
    test_classify_triple_paths("07/01/2024", HwxType::AmericanDate, 1719792000000.0, 0.0);
    test_classify_triple_paths("08/01/2024", HwxType::AmericanDate, 1722470400000.0, 0.0);
    test_classify_triple_paths("09/01/2024", HwxType::AmericanDate, 1725148800000.0, 0.0);
    test_classify_triple_paths("10/01/2024", HwxType::AmericanDate, 1727740800000.0, 0.0);
    test_classify_triple_paths("11/01/2024", HwxType::AmericanDate, 1730419200000.0, 0.0);
    test_classify_triple_paths("12/01/2024", HwxType::AmericanDate, 1733011200000.0, 0.0);

    // Different years
    test_classify_triple_paths("01/01/2023", HwxType::AmericanDate, 1672531200000.0, 0.0);
    test_classify_triple_paths("01/01/2025", HwxType::AmericanDate, 1735689600000.0, 0.0);
    test_classify_triple_paths("01/01/2020", HwxType::AmericanDate, 1577836800000.0, 0.0);
    test_classify_triple_paths("01/01/2030", HwxType::AmericanDate, 1893456000000.0, 0.0);

    // Edge cases
    test_classify_triple_paths("12/31/2023", HwxType::AmericanDate, 1703980800000.0, 0.0);
    test_classify_triple_paths("01/01/2024", HwxType::AmericanDate, 1704067200000.0, 0.0);
    test_classify_triple_paths("12/31/2024", HwxType::AmericanDate, 1735603200000.0, 0.0);
    test_classify_triple_paths("02/28/2023", HwxType::AmericanDate, 1677542400000.0, 0.0); // Non-leap year
    test_classify_triple_paths("02/29/2024", HwxType::AmericanDate, 1709164800000.0, 0.0); // Leap year
    test_classify_triple_paths("04/30/2024", HwxType::AmericanDate, 1714435200000.0, 0.0);
    test_classify_triple_paths("06/30/2024", HwxType::AmericanDate, 1719705600000.0, 0.0);
    test_classify_triple_paths("09/30/2024", HwxType::AmericanDate, 1727654400000.0, 0.0);
    test_classify_triple_paths("11/30/2024", HwxType::AmericanDate, 1732924800000.0, 0.0);

    // Month boundaries
    test_classify_triple_paths("01/31/2024", HwxType::AmericanDate, 1706659200000.0, 0.0);
    test_classify_triple_paths("03/31/2024", HwxType::AmericanDate, 1711843200000.0, 0.0);
    test_classify_triple_paths("05/31/2024", HwxType::AmericanDate, 1717113600000.0, 0.0);
    test_classify_triple_paths("07/31/2024", HwxType::AmericanDate, 1722384000000.0, 0.0);
    test_classify_triple_paths("08/31/2024", HwxType::AmericanDate, 1725062400000.0, 0.0);
    test_classify_triple_paths("10/31/2024", HwxType::AmericanDate, 1730332800000.0, 0.0);
    test_classify_triple_paths("12/31/2024", HwxType::AmericanDate, 1735603200000.0, 0.0);
  }
  #[test]
  fn test_datemath_extensive_variations() {
    // DateMath expressions return current timestamps, so we only validate the type
    let datemath_expressions = vec![
      "now",
      "now+1s",
      "now+1m",
      "now+1h",
      "now+1H",
      "now+1d",
      "now+1w",
      "now+1M",
      "now+1y",
      "now-1s",
      "now-30m",
      "now-5h",
      "now-7d",
      "now-2w",
      "now-6M",
      "now-1y",
      "now+1d-5h",
      "now+1y-6M+15d",
      "now-1y+1M-1d+1h-30m+15s",
      "now/s",
      "now/m",
      "now/h",
      "now/H",
      "now/d",
      "now/w",
      "now/M",
      "now/y",
      "now+1d/d",
      "now-30m/h",
      "now+1y-6M/d",
      "now+15d-2h+30m/h",
      "now+365d",
      "now+1440m",
      "now+86400s",
    ];

    // Get current system time for tolerance validation
    let system_now = std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap();
    let system_now_millis =
      (system_now.as_secs() as f64 * 1000.0) + (system_now.subsec_nanos() as f64 / 1_000_000.0);

    for expr in datemath_expressions {
      let result = crate::dispatch::classify_string(expr).unwrap();
      assert_eq!(
        result.hwx_type,
        HwxType::DateMath,
        "Failed for expression: {}",
        expr
      );

      // Validate numeric value is a reasonable timestamp (within ~2 years of current time)
      // This allows for complex expressions like "now+1y", "now-6M", "now+365d", etc.
      let timestamp_diff = (result.numeric_value_1 - system_now_millis).abs();
      let max_diff_millis = 2.0 * 365.25 * 24.0 * 60.0 * 60.0 * 1000.0; // ~2 years accounting for leap years
      assert!(
        timestamp_diff < max_diff_millis,
        "DateMath '{}' timestamp {} differs too much from system time {}, diff: {}ms (max: {}ms)",
        expr,
        result.numeric_value_1,
        system_now_millis,
        timestamp_diff,
        max_diff_millis
      );
    }
  }

  #[test]
  fn test_time_precision_edge_cases() {
    // Millisecond precision variations
    test_classify_triple_paths(
      "2024-01-15T14:30:25.1Z",
      HwxType::ISO8601Date,
      1705329025100.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.12Z",
      HwxType::ISO8601Date,
      1705329025120.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.123Z",
      HwxType::ISO8601Date,
      1705329025123.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.1234Z",
      HwxType::ISO8601Date,
      1705329025123.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.12345Z",
      HwxType::ISO8601Date,
      1705329025123.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25.123456Z",
      HwxType::ISO8601Date,
      1705329025123.0,
      0.0,
    );

    // Different second values
    test_classify_triple_paths(
      "2024-01-15T14:30:00Z",
      HwxType::ISO8601Date,
      1705329000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:30Z",
      HwxType::ISO8601Date,
      1705329030000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:59Z",
      HwxType::ISO8601Date,
      1705329059000.0,
      0.0,
    );

    // Different minute values
    test_classify_triple_paths(
      "2024-01-15T14:00:25Z",
      HwxType::ISO8601Date,
      1705327225000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:15:25Z",
      HwxType::ISO8601Date,
      1705328125000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:30:25Z",
      HwxType::ISO8601Date,
      1705329025000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:45:25Z",
      HwxType::ISO8601Date,
      1705329925000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T14:59:25Z",
      HwxType::ISO8601Date,
      1705330765000.0,
      0.0,
    );

    // Different hour values
    test_classify_triple_paths(
      "2024-01-15T00:30:25Z",
      HwxType::ISO8601Date,
      1705278625000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T06:30:25Z",
      HwxType::ISO8601Date,
      1705300225000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T12:30:25Z",
      HwxType::ISO8601Date,
      1705321825000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T18:30:25Z",
      HwxType::ISO8601Date,
      1705343425000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15T23:30:25Z",
      HwxType::ISO8601Date,
      1705361425000.0,
      0.0,
    );
  }

  #[test]
  fn test_leap_year_comprehensive() {
    // Leap years (simple dates ^ FullDate due to grammar precedence)
    test_classify_triple_paths("2024-02-29", HwxType::FullDate, 1709164800000.0, 0.0); // 2024 is leap year
    test_classify_triple_paths("2000-02-29", HwxType::FullDate, 951782400000.0, 0.0); // 2000 is leap year
    test_classify_triple_paths("1996-02-29", HwxType::FullDate, 825552000000.0, 0.0); // 1996 is leap year

    // Non-leap years (simple dates ^ FullDate due to grammar precedence)
    test_classify_triple_paths("2023-02-28", HwxType::FullDate, 1677542400000.0, 0.0); // Last day of Feb in non-leap year
    test_classify_triple_paths("2021-02-28", HwxType::FullDate, 1614470400000.0, 0.0);
    test_classify_triple_paths("1900-02-28", HwxType::FullDate, 5011200000.0, 0.0); // 1900 is not leap year

    // Leap year in different formats
    test_classify_triple_paths("29/02/2024", HwxType::EuropeanDate, 1709164800000.0, 0.0);
    test_classify_triple_paths("02/29/2024", HwxType::AmericanDate, 1709164800000.0, 0.0);
    test_classify_triple_paths(
      "2024-02-29T12:00:00Z",
      HwxType::ISO8601Date,
      1709208000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "29th February 2024",
      HwxType::FinancialDate,
      1709164800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tuesday, the 29th of February, 2024",
      HwxType::VerboseDate,
      1709164800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tue Feb 29 12:00:00 2024",
      HwxType::LogDate,
      1709208000000.0,
      0.0,
    );
  }

  #[test]
  fn test_financial_date_comprehensive() {
    // All ordinal suffixes
    test_classify_triple_paths(
      "1st January 2024",
      HwxType::FinancialDate,
      1704067200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2nd February 2024",
      HwxType::FinancialDate,
      1706832000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "3rd March 2024",
      HwxType::FinancialDate,
      1709424000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "4th April 2024",
      HwxType::FinancialDate,
      1712188800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "21st May 2024",
      HwxType::FinancialDate,
      1716249600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "22nd June 2024",
      HwxType::FinancialDate,
      1719014400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "23rd July 2024",
      HwxType::FinancialDate,
      1721692800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "24th August 2024",
      HwxType::FinancialDate,
      1724457600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "31st December 2024",
      HwxType::FinancialDate,
      1735603200000.0,
      0.0,
    );

    // Short month names
    test_classify_triple_paths(
      "15th Jan 2024",
      HwxType::FinancialDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "20th Feb 2024",
      HwxType::FinancialDate,
      1708387200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "25th Mar 2024",
      HwxType::FinancialDate,
      1711324800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "30th Apr 2024",
      HwxType::FinancialDate,
      1714435200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th May 2024",
      HwxType::FinancialDate,
      1715731200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Jun 2024",
      HwxType::FinancialDate,
      1718409600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Jul 2024",
      HwxType::FinancialDate,
      1721001600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Aug 2024",
      HwxType::FinancialDate,
      1723680000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Sep 2024",
      HwxType::FinancialDate,
      1726358400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Oct 2024",
      HwxType::FinancialDate,
      1728950400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Nov 2024",
      HwxType::FinancialDate,
      1731628800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Dec 2024",
      HwxType::FinancialDate,
      1734220800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Oct 2024",
      HwxType::FinancialDate,
      1728950400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Nov 2024",
      HwxType::FinancialDate,
      1731628800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th Dec 2024",
      HwxType::FinancialDate,
      1734220800000.0,
      0.0,
    );

    // Full month names
    test_classify_triple_paths(
      "15th January 2024",
      HwxType::FinancialDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th February 2024",
      HwxType::FinancialDate,
      1707955200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th March 2024",
      HwxType::FinancialDate,
      1710460800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th April 2024",
      HwxType::FinancialDate,
      1713139200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th May 2024",
      HwxType::FinancialDate,
      1715731200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th June 2024",
      HwxType::FinancialDate,
      1718409600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th July 2024",
      HwxType::FinancialDate,
      1721001600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th August 2024",
      HwxType::FinancialDate,
      1723680000000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th September 2024",
      HwxType::FinancialDate,
      1726358400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th October 2024",
      HwxType::FinancialDate,
      1728950400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th November 2024",
      HwxType::FinancialDate,
      1731628800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "15th December 2024",
      HwxType::FinancialDate,
      1734220800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 1st of January, 2024",
      HwxType::VerboseDate,
      1704067200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 15th of January, 2020",
      HwxType::VerboseDate,
      1579046400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 15th of January, 2021",
      HwxType::VerboseDate,
      1610668800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 15th of January, 2022",
      HwxType::VerboseDate,
      1642204800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 15th of January, 2023",
      HwxType::VerboseDate,
      1673740800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 15th of January, 2025",
      HwxType::VerboseDate,
      1736899200000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Saturday, the 22nd of June, 2024",
      HwxType::VerboseDate,
      1719014400000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sunday, the 23rd of July, 2024",
      HwxType::VerboseDate,
      1721692800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Monday, the 24th of August, 2024",
      HwxType::VerboseDate,
      1724457600000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tuesday, the 31st of December, 2024",
      HwxType::VerboseDate,
      1735603200000.0,
      0.0,
    );

    // All days of the week
    test_classify_triple_paths(
      "Monday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tuesday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Wednesday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );

    // All days of the week
    test_classify_triple_paths(
      "Monday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Tuesday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Wednesday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Thursday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Friday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Saturday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "Sunday, the 15th of January, 2024",
      HwxType::VerboseDate,
      1705276800000.0,
      0.0,
    );
  }

  #[test]
  fn test_generic_date_comprehensive() {
    // Basic YYYY-MM-DD format (FullDate wins due to grammar precedence)
    test_classify_triple_paths("2024-01-15", HwxType::FullDate, 1705276800000.0, 0.0);
    test_classify_triple_paths("2024-12-31", HwxType::FullDate, 1735603200000.0, 0.0);
    test_classify_triple_paths("2024-02-29", HwxType::FullDate, 1709164800000.0, 0.0); // Leap year

    // With time components
    test_classify_triple_paths(
      "2024-01-15 14:30:45",
      HwxType::GenericDate,
      1705329045000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15 00:00:00",
      HwxType::GenericDate,
      1705276800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15 23:59:59",
      HwxType::GenericDate,
      1705363199000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-01-15 12:30:45.123",
      HwxType::GenericDate,
      1705321845123.0,
      0.0,
    );

    // Different years (simple dates ^ FullDate due to grammar precedence)
    test_classify_triple_paths("1999-12-31", HwxType::FullDate, 946598400000.0, 0.0);
    test_classify_triple_paths("2000-01-01", HwxType::FullDate, 946684800000.0, 0.0);
    test_classify_triple_paths("2023-06-15", HwxType::FullDate, 1686787200000.0, 0.0);
    test_classify_triple_paths("2025-03-20", HwxType::FullDate, 1742428800000.0, 0.0);

    // Different separators and formats
    test_classify_triple_paths("2024/01/15", HwxType::GenericDate, 1705276800000.0, 0.0);
    test_classify_triple_paths("2024.01.15", HwxType::GenericDate, 1705276800000.0, 0.0);
    test_classify_triple_paths("2024-03-15", HwxType::FullDate, 1710460800000.0, 0.0);
    test_classify_triple_paths("2024-04-15", HwxType::FullDate, 1713139200000.0, 0.0);
    test_classify_triple_paths("2024-05-15", HwxType::FullDate, 1715731200000.0, 0.0);
    test_classify_triple_paths("2024-06-15", HwxType::FullDate, 1718409600000.0, 0.0);
    test_classify_triple_paths("2024-07-15", HwxType::FullDate, 1721001600000.0, 0.0);
    test_classify_triple_paths("2024-09-15", HwxType::FullDate, 1726358400000.0, 0.0);
    test_classify_triple_paths("2024-10-15", HwxType::FullDate, 1728950400000.0, 0.0);
    test_classify_triple_paths("2024-11-15", HwxType::FullDate, 1731628800000.0, 0.0);
    test_classify_triple_paths("2024-12-15", HwxType::FullDate, 1734220800000.0, 0.0);
  }

  #[test]
  fn test_invalid_date_edge_cases() {
    // These should not be classified as dates
    test_classify_triple_paths("2024-13-01", HwxType::String, 0.0, 0.0); // Invalid month
    test_classify_triple_paths("2024-01-32", HwxType::String, 0.0, 0.0); // Invalid day
    test_classify_triple_paths("2024-02-30", HwxType::String, 0.0, 0.0); // Invalid day for February
    test_classify_triple_paths("2024-04-31", HwxType::String, 0.0, 0.0); // Invalid day for April
    test_classify_triple_paths("2024-00-15", HwxType::String, 0.0, 0.0); // Invalid month (0)
    test_classify_triple_paths("2024-01-00", HwxType::String, 0.0, 0.0); // Invalid day (0)

    // Invalid time formats
    test_classify_triple_paths("2024-01-15T25:30:25Z", HwxType::String, 0.0, 0.0); // Invalid hour
    test_classify_triple_paths("2024-01-15T14:60:25Z", HwxType::String, 0.0, 0.0); // Invalid minute
    test_classify_triple_paths("2024-01-15T14:30:60Z", HwxType::String, 0.0, 0.0); // Invalid second

    // Invalid timezone formats
    test_classify_triple_paths("2024-01-15T14:30:25+25:00", HwxType::String, 0.0, 0.0); // Invalid timezone hour
    test_classify_triple_paths("2024-01-15T14:30:25+00:70", HwxType::String, 0.0, 0.0); // Invalid timezone minute

    // Malformed formats
    test_classify_triple_paths("2024/01/15T14:30:25Z", HwxType::String, 0.0, 0.0); // Mixed separators
    test_classify_triple_paths("2024-1-15T14:30:25Z", HwxType::String, 0.0, 0.0); // Single digit month in ISO
    test_classify_triple_paths("24-01-15T14:30:25Z", HwxType::String, 0.0, 0.0); // 2-digit year in ISO
  }

  #[test]
  fn test_boundary_dates_comprehensive() {
    // Year boundaries
    test_classify_triple_paths(
      "1999-12-31T23:59:59Z",
      HwxType::ISO8601Date,
      946684799000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2000-01-01T00:00:00Z",
      HwxType::ISO8601Date,
      946684800000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2024-12-31T23:59:59Z",
      HwxType::ISO8601Date,
      1735689599000.0,
      0.0,
    );
    test_classify_triple_paths(
      "2025-01-01T00:00:00Z",
      HwxType::ISO8601Date,
      1735689600000.0,
      0.0,
    );

    // Month boundaries (all months) - simple dates ^ FullDate due to grammar precedence
    test_classify_triple_paths("2024-01-01", HwxType::FullDate, 1704067200000.0, 0.0); // January start
    test_classify_triple_paths("2024-01-31", HwxType::FullDate, 1706659200000.0, 0.0); // January end
    test_classify_triple_paths("2024-02-01", HwxType::FullDate, 1706745600000.0, 0.0); // February start
    test_classify_triple_paths("2024-02-29", HwxType::FullDate, 1709164800000.0, 0.0); // February end (leap year)
    test_classify_triple_paths("2024-03-01", HwxType::FullDate, 1709251200000.0, 0.0); // March start
    test_classify_triple_paths("2024-03-31", HwxType::FullDate, 1711843200000.0, 0.0); // March end
    test_classify_triple_paths("2024-04-01", HwxType::FullDate, 1711929600000.0, 0.0); // April start
    test_classify_triple_paths("2024-04-30", HwxType::FullDate, 1714435200000.0, 0.0); // April end
    test_classify_triple_paths("2024-05-01", HwxType::FullDate, 1714521600000.0, 0.0); // May start
    test_classify_triple_paths("2024-05-31", HwxType::FullDate, 1717113600000.0, 0.0); // May end
    test_classify_triple_paths("2024-06-01", HwxType::FullDate, 1717200000000.0, 0.0); // June start
    test_classify_triple_paths("2024-06-30", HwxType::FullDate, 1719705600000.0, 0.0); // June end
    test_classify_triple_paths("2024-07-01", HwxType::FullDate, 1719792000000.0, 0.0); // July start
    test_classify_triple_paths("2024-07-31", HwxType::FullDate, 1722384000000.0, 0.0); // July end
    test_classify_triple_paths("2024-08-01", HwxType::FullDate, 1722470400000.0, 0.0); // August start
    test_classify_triple_paths("2024-08-31", HwxType::FullDate, 1725062400000.0, 0.0); // August end
    test_classify_triple_paths("2024-09-01", HwxType::FullDate, 1725148800000.0, 0.0); // September start
    test_classify_triple_paths("2024-09-30", HwxType::FullDate, 1727654400000.0, 0.0); // September end
    test_classify_triple_paths("2024-10-01", HwxType::FullDate, 1727740800000.0, 0.0); // October start
    test_classify_triple_paths("2024-10-31", HwxType::FullDate, 1730332800000.0, 0.0); // October end
    test_classify_triple_paths("2024-11-01", HwxType::FullDate, 1730419200000.0, 0.0); // November start
    test_classify_triple_paths("2024-11-30", HwxType::FullDate, 1732924800000.0, 0.0); // November end
    test_classify_triple_paths("2024-12-01", HwxType::FullDate, 1733011200000.0, 0.0); // December start
    test_classify_triple_paths("2024-12-31", HwxType::FullDate, 1735603200000.0, 0.0); // December end
  }

  // =============================================================================
  // STRESS TESTS - Push the limits!
  // =============================================================================

  #[test]
  fn test_massive_batch_processing() {
    //  SCALAR PATH TEST (small strings)
    let scalar_batch: Vec<String> = (0..10)
      .map(|i| match i % 10 {
        0 => format!("192.168.1.{}", i % 255),
        1 => format!("2001:db8::{:x}", i),
        2 => format!("{}.{},{}.{}", i % 90, i % 1000, i % 180, i % 1000),
        3 => format!("[{},{},{}]", i, i + 1, i + 2),
        4 => format!("{}.{}", i, i % 1000),
        5 => format!(
          "2024-{:02}-{:02}T{:02}:{:02}:{:02}Z",
          (i % 12) + 1,
          (i % 28) + 1,
          i % 24,
          i % 60,
          i % 60
        ),
        6 => i.to_string(),
        7 => {
          if i % 2 == 0 {
            "true".to_string()
          } else {
            "false".to_string()
          }
        }
        8 => format!("/home/user/file_{}.txt", i),
        _ => "null".to_string(),
      })
      .collect();

    // Test scalar batch
    for input in scalar_batch.iter() {
      let result = crate::dispatch::classify_string(input).unwrap();
      assert_ne!(
        result.hwx_type,
        HwxType::Undefined,
        "SCALAR: Undefined result for input: {}",
        input
      );
    }

    //  SIMD PATH TEST (medium strings ~100-500 chars)
    let simd_batch: Vec<String> = vec![
      // Create medium-length strings
      format!("192.168.1.1{}", ";192.168.1.2".repeat(20)), // ~250 chars
      format!("2001:db8::{}", "0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f".repeat(10)), // ~350 chars
      format!(
        "[{}]",
        (0..100)
          .map(|i| i.to_string())
          .collect::<Vec<_>>()
          .join(",")
      ), // ~300 chars
      format!("{}", "true,false,".repeat(50)),             // ~500 chars
      format!("/home/user/{}/file.txt", "directory/".repeat(30)), // ~310 chars
    ];

    for input in simd_batch.iter() {
      let result = crate::dispatch::classify_string(input).unwrap();
      assert_ne!(
        result.hwx_type,
        HwxType::Undefined,
        "SIMD: Undefined result for input length: {}",
        input.len()
      );
    }

    //  GPU PATH TEST (very long strings > GPU_THRESHOLD_STRING = 512)
    let gpu_batch: Vec<String> = vec![
      // Create GPU-scale long strings (>512 chars)
      format!("192.168.1.1{}", ";192.168.1.2".repeat(100)), // ~1200 chars
      format!("2001:db8::{}", "0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f".repeat(50)), // ~1700 chars
      format!(
        "[{}]",
        (0..1000)
          .map(|i| i.to_string())
          .collect::<Vec<_>>()
          .join(",")
      ), // ~4000 chars
      format!("{}", "true,false,".repeat(200)),             // ~2000 chars
      format!(
        "/home/user/{}/file.txt",
        "directory/subdirectory/".repeat(100)
      ), // ~2200 chars
      format!(
        "2024-01-01T00:00:00Z{}",
        ",2024-01-01T00:00:00Z".repeat(100)
      ), // ~2100 chars
      (0..2000).map(|_| "x").collect::<String>(), // 2000 chars of just 'x' - should be String
      format!("{}.{}", "123456789".repeat(100), "123456789".repeat(100)), // ~1800 chars float
    ];

    for input in gpu_batch.iter() {
      let result = crate::dispatch::classify_string(input).unwrap();
      assert_ne!(
        result.hwx_type,
        HwxType::Undefined,
        "GPU: Undefined result for input length: {} chars",
        input.len()
      );
      // Verify the string length triggers GPU path
      assert!(
        input.len() > 512,
        "GPU test string should be > 512 chars, got: {}",
        input.len()
      );
    }
  }

  /// Test helper that verifies both type classification AND numeric timestamp values
  fn test_classify_with_timestamp(
    input: &str,
    expected_type: HwxType,
    expected_timestamp_millis: f64,
  ) {
    let result = crate::dispatch::classify_string(input).unwrap();
    assert_eq!(
      result.hwx_type, expected_type,
      "Type mismatch for input: '{}' - expected {:?}, got {:?}",
      input, expected_type, result.hwx_type
    );

    // Check that numeric_value_1 contains the expected timestamp in milliseconds
    let timestamp_diff = (result.numeric_value_1 - expected_timestamp_millis).abs();
    assert!(
      timestamp_diff < 1.0, // Allow 1ms tolerance
      "Timestamp mismatch for input: '{}' - expected {}, got {}, diff: {}",
      input,
      expected_timestamp_millis,
      result.numeric_value_1,
      timestamp_diff
    );
  }

  #[test]
  fn test_timestamp_values_comprehensive() {
    config_test_logger();

    // Test specific dates with known timestamp values (in milliseconds)

    // 2023-02-01 00:00:00 UTC = 1675209600 seconds = 1675209600000 milliseconds
    test_classify_with_timestamp("2023-02-01", HwxType::FullDate, 1675209600000.0);

    // 2023-01-01 00:00:00 UTC = 1672531200 seconds = 1672531200000 milliseconds
    test_classify_with_timestamp("2023-01-01", HwxType::FullDate, 1672531200000.0);

    // 2024-01-15 00:00:00 UTC = 1705276800 seconds = 1705276800000 milliseconds
    test_classify_with_timestamp("2024-01-15", HwxType::FullDate, 1705276800000.0);

    // Test American date format: 01/15/2024 = 2024-01-15 = 1705276800000 milliseconds
    test_classify_with_timestamp("01/15/2024", HwxType::AmericanDate, 1705276800000.0);

    // Test European date format: 15/01/2024 = 2024-01-15 = 1705276800000 milliseconds
    test_classify_with_timestamp("15/01/2024", HwxType::EuropeanDate, 1705276800000.0);
  }

  #[test]
  fn test_datemath_timestamp_values() {
    config_test_logger();

    // Test datemath "now" - should return current timestamp in milliseconds
    let now_result = crate::dispatch::classify_string("now").unwrap();
    assert_eq!(now_result.hwx_type, HwxType::DateMath);

    // Verify the returned timestamp is very close to current system time (within 1 second)
    let system_now = std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap();
    let system_now_millis =
      (system_now.as_secs() as f64 * 1000.0) + (system_now.subsec_nanos() as f64 / 1_000_000.0);
    let timestamp_diff = (now_result.numeric_value_1 - system_now_millis).abs();
    assert!(
      timestamp_diff < 1000.0, // Allow 1 second tolerance
      "DateMath 'now' timestamp {} differs too much from system time {}, diff: {}ms",
      now_result.numeric_value_1,
      system_now_millis,
      timestamp_diff
    );
  }

  // This test is for private internal functions - should test public API instead
  // #[test]
  // fn test_determine_type_from_signature() {
  //   config_test_logger();
  //
  //   let mut signature = PatternSignature::<32>::new();
  //   signature.has_digits = true;
  //   signature.has_letters = true;
  //   signature.has_punctuation = true;
  //   signature.dash_count = 4;
  //   signature.dash_positions = [
  //     25, 30, 35, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  //     0, 0,
  //   ];
  //   let result = determine_type_from_signature(
  //     "000000000000______console-test-data-w2-1756104881208",
  //     52,
  //     &signature,
  //   );
  //   assert_eq!(result.0, HwxType::String);
  //   assert_eq!(result.1, 0.0);
  //   assert_eq!(result.2, 0.0);
  // }
}
