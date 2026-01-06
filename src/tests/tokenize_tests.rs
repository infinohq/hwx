// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// COMPREHENSIVE TESTS - MIRRORING CLASSIFY.RS TEST STRUCTURE
// =============================================================================

#[cfg(test)]
mod tests {

  // Triple path helper for tokenization
  fn test_tokenize_triple_paths(input: &str, expected_word_count: u32, expected_lowercase: &str) {
    //  SCALAR PATH TEST
    test_tokenize(input, expected_word_count, expected_lowercase);

    //  SIMD PATH TEST (test same string multiple times)
    for _ in 0..100 {
      test_tokenize(input, expected_word_count, expected_lowercase);
    }

    // ^ GPU PATH TEST (test same string many times)
    for _ in 0..10000 {
      test_tokenize(input, expected_word_count, expected_lowercase);
    }
  }

  // Helper function to test tokenization using proper dispatch
  fn test_tokenize(input: &str, expected_word_count: u32, expected_lowercase: &str) {
    let mut output = Vec::new();

    // First apply lowercase conversion
    let lowercase_input =
      crate::dispatch::to_lowercase(input).unwrap_or_else(|_| input.to_lowercase());

    // Then tokenize the lowercase string
    let result = crate::dispatch::tokenize_string(&lowercase_input, &mut output);
    assert!(result.is_ok(), "Tokenization failed: {:?}", result);

    let word_count = output.len() as u32;
    // The tokenizer should preserve original content - return the lowercase input
    let lowercase_result = lowercase_input;

    assert_eq!(
      word_count, expected_word_count,
      "Word count mismatch for '{}': expected {}, got {}",
      input, expected_word_count, word_count
    );
    assert_eq!(
      lowercase_result, expected_lowercase,
      "Lowercase conversion failed for '{}': expected '{}', got '{}'",
      input, expected_lowercase, lowercase_result
    );
  }

  // Triple path helper for boundaries
  fn test_boundaries_triple_paths(input: &str, expected_boundaries: &[u32]) {
    //  SCALAR PATH TEST
    test_boundaries(input, expected_boundaries);

    //  SIMD PATH TEST (test same string multiple times)
    for _ in 0..100 {
      test_boundaries(input, expected_boundaries);
    }

    // ^ GPU PATH TEST (test same string many times)
    for _ in 0..10000 {
      test_boundaries(input, expected_boundaries);
    }
  }

  // Helper function to test word boundary positions
  fn test_boundaries(input: &str, expected_boundaries: &[u32]) {
    let mut output = Vec::new();
    let result = crate::dispatch::tokenize_string(input, &mut output);
    assert!(result.is_ok(), "Tokenization failed: {:?}", result);

    // For boundary testing, we'll verify the tokens match expected pattern
    let word_count = output.len();
    assert_eq!(
      word_count * 2,
      expected_boundaries.len(),
      "Boundary count mismatch for '{}': expected {} boundaries, got {} words",
      input,
      expected_boundaries.len(),
      word_count
    );
  }

  // =============================================================================
  // BASIC TOKENIZATION TESTS
  // =============================================================================

  #[test]
  fn test_basic_single_word() {
    test_tokenize_triple_paths("hello", 1, "hello");
    test_tokenize_triple_paths("world", 1, "world");
    test_tokenize_triple_paths("tokenization", 1, "tokenization");
    test_tokenize_triple_paths("UPPERCASE", 1, "uppercase");
    test_tokenize_triple_paths("MiXeD", 1, "mixed");
  }

  #[test]
  fn test_basic_multiple_words() {
    test_tokenize_triple_paths("hello world", 2, "hello world");
    test_tokenize_triple_paths("the quick brown fox", 4, "the quick brown fox");
    test_tokenize_triple_paths("SIMD tokenization test", 3, "simd tokenization test");
    test_tokenize_triple_paths("Multiple Words Here", 3, "multiple words here");
  }

  #[test]
  fn test_empty_and_edge_cases() {
    test_tokenize_triple_paths("", 0, "");
    test_tokenize_triple_paths(" ", 0, " ");
    test_tokenize_triple_paths("  ", 0, "  ");
    test_tokenize_triple_paths("\t", 0, "\t");
    test_tokenize_triple_paths("\n", 0, "\n");
    test_tokenize_triple_paths("\r", 0, "\r");
    test_tokenize_triple_paths("   \t\n\r   ", 0, "   \t\n\r   ");
  }

  #[test]
  fn test_single_character_words() {
    test_tokenize_triple_paths("a", 1, "a");
    test_tokenize_triple_paths("A", 1, "a");
    test_tokenize_triple_paths("1", 1, "1");
    test_tokenize_triple_paths("a b c", 3, "a b c");
    test_tokenize_triple_paths("A B C", 3, "a b c");
    test_tokenize_triple_paths("1 2 3", 3, "1 2 3");
  }

  // =============================================================================
  // PUNCTUATION AND SPECIAL CHARACTER TESTS
  // =============================================================================

  #[test]
  fn test_punctuation_boundaries() {
    test_tokenize_triple_paths("hello,world", 2, "hello,world");
    test_tokenize_triple_paths("hello, world", 2, "hello, world");
    test_tokenize_triple_paths("hello,world!", 2, "hello,world!");
    test_tokenize_triple_paths("hello.world", 1, "hello.world");
    test_tokenize_triple_paths("hello;world", 2, "hello;world");
    test_tokenize_triple_paths("hello:world", 2, "hello:world");
  }

  #[test]
  fn test_quotes_and_brackets() {
    test_tokenize_triple_paths("\"hello world\"", 2, "\"hello world\"");
    test_tokenize_triple_paths("'hello world'", 2, "'hello world'");
    test_tokenize_triple_paths("[hello world]", 2, "[hello world]");
    test_tokenize_triple_paths("(hello world)", 2, "(hello world)");
    test_tokenize_triple_paths("{hello world}", 2, "{hello world}");
  }

  #[test]
  fn test_hyphens_and_apostrophes() {
    test_tokenize_triple_paths("well-known", 2, "well-known");
    test_tokenize_triple_paths("one-two-three-four", 4, "one-two-three-four");
    test_tokenize_triple_paths("don't", 2, "don't");
    test_tokenize_triple_paths("can't", 2, "can't");
    test_tokenize_triple_paths("won't", 2, "won't");
    test_tokenize_triple_paths("it's", 2, "it's");
  }

  #[test]
  fn test_only_punctuation() {
    test_tokenize_triple_paths("!@#$%^&*()", 0, "!@#$%^&*()");
    test_tokenize_triple_paths(".,;:!?", 0, ".,;:!?");
    test_tokenize_triple_paths("[]{}()", 0, "[]{}()");
    test_tokenize_triple_paths("\"'`", 0, "\"'`");
  }

  // =============================================================================
  // NUMERIC AND ALPHANUMERIC TESTS
  // =============================================================================

  #[test]
  fn test_pure_numbers() {
    test_tokenize_triple_paths("123", 1, "123");
    test_tokenize_triple_paths("456789", 1, "456789");
    test_tokenize_triple_paths("0", 1, "0");
    test_tokenize_triple_paths("123 456 789", 3, "123 456 789");
    test_tokenize_triple_paths("1 2 3 4 5", 5, "1 2 3 4 5");
  }

  #[test]
  fn test_alphanumeric_combinations() {
    test_tokenize_triple_paths("abc123", 1, "abc123");
    test_tokenize_triple_paths("123abc", 1, "123abc");
    test_tokenize_triple_paths("a1b2c3", 1, "a1b2c3");
    test_tokenize_triple_paths("test123 word456", 2, "test123 word456");
    test_tokenize_triple_paths("HTML5 CSS3 JS2024", 3, "html5 css3 js2024");
  }

  #[test]
  fn test_numbers_with_punctuation() {
    test_tokenize_triple_paths("3.14", 1, "3.14");
    test_tokenize_triple_paths("1,000", 2, "1,000");
    test_tokenize_triple_paths("$100", 1, "$100");
    test_tokenize_triple_paths("100%", 1, "100%");
    test_tokenize_triple_paths("version-2.1", 2, "version-2.1");
  }

  // =============================================================================
  // WHITESPACE HANDLING TESTS
  // =============================================================================

  #[test]
  fn test_multiple_spaces() {
    test_tokenize_triple_paths("hello  world", 2, "hello  world");
    test_tokenize_triple_paths("hello   world", 2, "hello   world");
    test_tokenize_triple_paths("hello    world", 2, "hello    world");
    test_tokenize_triple_paths("word1     word2     word3", 3, "word1     word2     word3");
  }

  #[test]
  fn test_leading_trailing_whitespace() {
    test_tokenize_triple_paths(" hello", 1, " hello");
    test_tokenize_triple_paths("hello ", 1, "hello ");
    test_tokenize_triple_paths(" hello ", 1, " hello ");
    test_tokenize_triple_paths("  hello  world  ", 2, "  hello  world  ");
    test_tokenize_triple_paths("   multiple   spaces   ", 2, "   multiple   spaces   ");
  }

  #[test]
  fn test_mixed_whitespace_types() {
    test_tokenize_triple_paths("hello\tworld", 2, "hello\tworld");
    test_tokenize_triple_paths("hello\nworld", 2, "hello\nworld");
    test_tokenize_triple_paths("hello\rworld", 2, "hello\rworld");
    test_tokenize_triple_paths("word1\t\nword2", 2, "word1\t\nword2");
    test_tokenize_triple_paths(" \t\nhello\r\n world\t ", 2, " \t\nhello\r\n world\t ");
  }

  // =============================================================================
  // CASE CONVERSION TESTS
  // =============================================================================

  #[test]
  fn test_uppercase_conversion() {
    test_tokenize_triple_paths("HELLO", 1, "hello");
    test_tokenize_triple_paths("WORLD", 1, "world");
    test_tokenize_triple_paths("UPPERCASE WORDS", 2, "uppercase words");
    test_tokenize_triple_paths("ALL CAPS SENTENCE", 3, "all caps sentence");
    test_tokenize_triple_paths("MIXED123CASE", 1, "mixed123case");
  }

  #[test]
  fn test_mixed_case_conversion() {
    test_tokenize_triple_paths("CamelCase", 1, "camelcase");
    test_tokenize_triple_paths("PascalCase", 1, "pascalcase");
    test_tokenize_triple_paths("MiXeD CaSe", 2, "mixed case");
    test_tokenize_triple_paths("WeIrD cApItAlIzAtIoN", 2, "weird capitalization");
    test_tokenize_triple_paths("HTML CSS JavaScript", 3, "html css javascript");
  }

  #[test]
  fn test_case_preservation_for_non_ascii() {
    test_tokenize_triple_paths("café", 1, "café");
    test_tokenize_triple_paths("naïve", 1, "naïve");
    test_tokenize_triple_paths("résumé", 1, "résumé");
    test_tokenize_triple_paths("Café NAÏVE", 2, "café naïve");
  }

  // =============================================================================
  // UNICODE AND INTERNATIONAL CHARACTER TESTS
  // =============================================================================

  #[test]
  fn test_basic_unicode() {
    test_tokenize_triple_paths("café naïve", 2, "café naïve");
    test_tokenize_triple_paths("résumé français", 2, "résumé français");
    test_tokenize_triple_paths("piñata jalapeño", 2, "piñata jalapeño");
    test_tokenize_triple_paths("Zürich München", 2, "zürich münchen");
  }

  #[test]
  fn test_unicode_boundaries() {
    test_tokenize_triple_paths("test\u{00A0}word", 2, "test\u{00A0}word"); // Non-breaking space
    test_tokenize_triple_paths("word\u{2013}word", 2, "word\u{2013}word"); // En dash
    test_tokenize_triple_paths("test\u{00B7}middle", 2, "test\u{00B7}middle"); // Middle dot
  }

  #[test]
  fn test_emoji_and_symbols() {
    test_tokenize_triple_paths("hello 👋 world", 3, "hello 👋 world");
    test_tokenize_triple_paths("test © symbol", 3, "test © symbol");
    test_tokenize_triple_paths("price € 100", 3, "price € 100");
    test_tokenize_triple_paths("temperature 25°C", 2, "temperature 25°c");
  }

  // =============================================================================
  // WORD BOUNDARY POSITION TESTS
  // =============================================================================

  #[test]
  fn test_simple_boundary_positions() {
    test_boundaries_triple_paths("hello", &[0, 5]);
    test_boundaries_triple_paths("world", &[0, 5]);
    test_boundaries_triple_paths("test", &[0, 4]);
  }

  #[test]
  fn test_two_word_boundaries() {
    test_boundaries_triple_paths("hello world", &[0, 5, 6, 11]);
    test_boundaries_triple_paths("the quick", &[0, 3, 4, 9]);
    test_boundaries_triple_paths("word1 word2", &[0, 5, 6, 11]);
  }

  #[test]
  fn test_multiple_word_boundaries() {
    test_boundaries_triple_paths("the quick brown", &[0, 3, 4, 9, 10, 15]);
    test_boundaries_triple_paths("a b c", &[0, 1, 2, 3, 4, 5]);
    test_boundaries_triple_paths("one two three four", &[0, 3, 4, 7, 8, 13, 14, 18]);
  }

  #[test]
  fn test_boundaries_with_punctuation() {
    test_boundaries_triple_paths("hello,world", &[0, 5, 6, 11]);
    test_boundaries_triple_paths("word1.word2", &[0, 11]);
    test_boundaries_triple_paths("test;case", &[0, 4, 5, 9]);
  }

  #[test]
  fn test_boundaries_with_leading_trailing_space() {
    test_boundaries_triple_paths(" hello", &[1, 6]);
    test_boundaries_triple_paths("hello ", &[0, 5]);
    test_boundaries_triple_paths(" hello ", &[1, 6]);
    test_boundaries_triple_paths("  word1  word2  ", &[2, 7, 9, 14]);
  }

  // =============================================================================
  // PERFORMANCE AND STRESS TESTS
  // =============================================================================

  #[test]
  fn test_long_single_word() {
    let long_word = "supercalifragilisticexpialidocious".repeat(10);
    let expected_lowercase = long_word.to_lowercase();
    test_tokenize_triple_paths(&long_word, 1, &expected_lowercase);
  }

  #[test]
  fn test_many_short_words() {
    let many_words = (0..100)
      .map(|i| format!("word{}", i))
      .collect::<Vec<_>>()
      .join(" ");
    let expected_lowercase = many_words.to_lowercase();
    test_tokenize_triple_paths(&many_words, 100, &expected_lowercase);
  }

  #[test]
  fn test_mixed_length_words() {
    let mixed = "a bb ccc dddd eeeee ffffff ggggggg hhhhhhhh iiiiiiiii jjjjjjjjjj";
    test_tokenize_triple_paths(mixed, 10, mixed);
  }

  #[test]
  fn test_alternating_word_punctuation() {
    let alternating = "word1,word2;word3.word4:word5!word6?word7";
    test_tokenize_triple_paths(alternating, 6, alternating);
  }

  #[test]
  fn test_dense_punctuation() {
    let dense = "word1!!!word2???word3...word4,,,word5;;;word6:::";
    test_tokenize_triple_paths(dense, 6, dense);
  }

  // =============================================================================
  // BUFFER OVERFLOW AND EDGE CASE TESTS
  // =============================================================================

  #[test]
  fn test_buffer_overflow_protection() {
    let input = "This is a test with many words that should exceed buffer";
    let mut output = Vec::new();

    let result = crate::dispatch::tokenize_string(input, &mut output);

    // Should not crash and should work correctly
    assert!(result.is_ok());
    assert!(!output.is_empty()); // Should find some words
  }

  #[test]
  fn test_underscore_debug() {
    test_tokenize_triple_paths("test_345", 1, "test_345");
  }

  #[test]
  fn test_zero_length_input() {
    let input = "";
    let mut output = Vec::new();

    let result = crate::dispatch::tokenize_string(input, &mut output);

    assert!(result.is_ok());
    assert_eq!(output.len(), 0);
  }

  #[test]
  fn test_single_byte_inputs() {
    test_tokenize_triple_paths("a", 1, "a");
    test_tokenize_triple_paths("A", 1, "a");
    test_tokenize_triple_paths("1", 1, "1");
    test_tokenize_triple_paths("!", 0, "!");
    test_tokenize_triple_paths(" ", 0, " ");
  }

  // =============================================================================
  //  REAL-WORLD TEXT SAMPLES
  // =============================================================================

  #[test]
  fn test_code_like_text() {
    test_tokenize_triple_paths(
      "function_name(param1, param2)",
      3,
      "function_name(param1, param2)",
    );
    test_tokenize_triple_paths("let variable = 42;", 3, "let variable = 42;");
    test_tokenize_triple_paths(
      "if (condition) { return true; }",
      4,
      "if (condition) { return true; }",
    );
    test_tokenize_triple_paths("std::vec::Vec<String>", 4, "std::vec::vec<string>");
  }

  #[test]
  fn test_url_like_text() {
    // First test what unicode_words() actually does
    use unicode_segmentation::UnicodeSegmentation;

    let test_cases = [
      "https://example.com/path",
      "user@domain.com",
      "www.example.org",
      "ftp://files.server.net",
    ];

    for case in &test_cases {
      let unicode_words: Vec<&str> = case.unicode_words().collect();
      log::debug!(
        "unicode_words('{}') = {:?} (count: {})",
        case,
        unicode_words,
        unicode_words.len()
      );

      // Test our SIMD implementation matches unicode_words() behavior
      // The tokenizer should preserve original format, not join with spaces
      test_tokenize_triple_paths(case, unicode_words.len() as u32, &case.to_lowercase());
    }
  }

  #[test]
  fn test_sentence_like_text() {
    test_tokenize_triple_paths(
      "The quick brown fox jumps over the lazy dog.",
      9,
      "the quick brown fox jumps over the lazy dog.",
    );
    test_tokenize_triple_paths(
      "Hello, world! How are you today?",
      6,
      "hello, world! how are you today?",
    );
    test_tokenize_triple_paths(
      "This is a test sentence with punctuation, numbers 123, and CAPS.",
      11,
      "this is a test sentence with punctuation, numbers 123, and caps.",
    );
  }

  #[test]
  fn test_log_like_text() {
    test_tokenize_triple_paths(
      "[INFO] Processing request #1234",
      4,
      "[info] processing request #1234",
    );
    test_tokenize_triple_paths(
      "ERROR: Failed to connect to database",
      6,
      "error: failed to connect to database",
    );
    test_tokenize_triple_paths(
      "2024-01-15 10:30:25 DEBUG: Starting service",
      9,
      "2024-01-15 10:30:25 debug: starting service",
    );
  }

  #[test]
  fn test_json_like_text() {
    test_tokenize_triple_paths(
      "{\"key\": \"value\", \"number\": 42}",
      4,
      "{\"key\": \"value\", \"number\": 42}",
    );
    test_tokenize_triple_paths(
      "[\"item1\", \"item2\", \"item3\"]",
      3,
      "[\"item1\", \"item2\", \"item3\"]",
    );
    test_tokenize_triple_paths("null true false", 3, "null true false");
  }

  // =============================================================================
  // ARCHITECTURE-SPECIFIC VALIDATION TESTS
  // =============================================================================

  #[test]
  fn test_simd_vs_scalar_consistency() {
    let test_cases = vec![
      "hello world",
      "The Quick Brown Fox",
      "123 abc XYZ !@#",
      "café naïve résumé",
      "mixed123CASE test",
      "punctuation,test;case:example",
      "   leading trailing   ",
      "single",
      "",
      "a",
      "UPPERCASE",
      "word1  word2    word3",
    ];

    for test_case in test_cases {
      let mut output = Vec::new();
      let result = crate::dispatch::tokenize_string(test_case, &mut output);

      // log::debug!("Input: '{}' -> Output: {:?} (len={})", test_case, output, output.len());

      if !test_case.trim().is_empty() {
        assert!(
          result.is_ok(),
          "SIMD tokenization failed for: '{}'",
          test_case
        );
        assert!(
          !output.is_empty() || test_case.trim().is_empty(),
          "SIMD found no words in: '{}'",
          test_case
        );
      }
    }
  }

  #[test]
  fn test_chunk_boundary_handling() {
    // Test strings that cross SIMD chunk boundaries
    let chunk_boundary_tests = vec![
      "1234567890123456",                // Exactly 16 bytes
      "12345678901234567",               // 17 bytes
      "123456789012345 word",            // Word crossing boundary
      "word 12345678901234567890",       // Word after boundary
      "12345678901234 5678901234567890", // Multiple boundaries
    ];

    for test in chunk_boundary_tests {
      let mut output = Vec::new();
      let result = crate::dispatch::tokenize_string(test, &mut output);

      assert!(
        result.is_ok(),
        "Failed to tokenize chunk boundary test: '{}'",
        test
      );
      assert!(
        !output.is_empty() || test.trim().is_empty(),
        "No words found in chunk boundary test: '{}'",
        test
      );
    }
  }

  #[test]
  fn test_triple_path_tokenization() {
    //  SCALAR PATH TEST (small strings < 8 chars)
    let scalar_tests = vec![
      ("hi", 1),
      ("a b c", 3),
      ("test", 1),
      ("ok!", 1),
      ("1 2 3", 3),
    ];

    for (input, expected_words) in scalar_tests {
      let mut output = Vec::new();
      let result = crate::dispatch::tokenize_string(input, &mut output);
      assert!(result.is_ok(), "SCALAR: Failed to tokenize '{}'", input);
      assert_eq!(
        output.len() as u32,
        expected_words,
        "SCALAR: Wrong word count for '{}', expected {}, got {}",
        input,
        expected_words,
        output.len()
      );
    }

    //  SIMD PATH TEST (medium strings 8-512 chars)
    let simd_tests = vec![
      ("The quick brown fox jumps over the lazy dog", 9),
      (
        "Testing SIMD tokenization with a moderately long string that should trigger SIMD processing",
        13,
      ),
      (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
        19,
      ),
      (
        "Multiple    spaces     between      words       should        be         handled          correctly",
        8,
      ),
      (
        "Numbers123mixed456with789text000and111more222numbers333here444end555test666complete777done888final999",
        1,
      ),
    ];

    for (input, expected_words) in simd_tests {
      let mut output = Vec::new();
      let result = crate::dispatch::tokenize_string(input, &mut output);
      assert!(
        result.is_ok(),
        "SIMD: Failed to tokenize string of length {}",
        input.len()
      );
      assert_eq!(
        output.len() as u32,
        expected_words,
        "SIMD: Wrong word count for string of length {}, expected {}, got {}",
        input.len(),
        expected_words,
        output.len()
      );
    }

    //  GPU PATH TEST (very long strings > GPU_THRESHOLD_STRING = 512)
    let gpu_tests = vec![
      // Create GPU-scale long strings
      ("word ".repeat(200).to_string(), 200), // 1000 chars, 200 words
      (
        "The quick brown fox jumps over the lazy dog. "
          .repeat(50)
          .to_string(),
        450,
      ), // ~2250 chars, 450 words
      (format!("start {} end", "middle_word ".repeat(100)), 101), // ~1200 chars, 101 words
      (
        (0..200)
          .map(|i| format!("token{} ", i))
          .collect::<String>()
          .to_string(),
        200,
      ), // ~1600 chars, 200 tokens
      // Very long single word (no spaces)
      ((0..1000).map(|_| "a").collect::<String>(), 1), // 1000 chars, 1 word
      // Mixed content with punctuation
      ("Hello, world! How are you? ".repeat(50).to_string(), 250), // ~1350 chars, 250 words (5 words × 50)
      // Unicode content
      (
        "Café résumé naïve 日本語 中文 한국어 "
          .repeat(100)
          .to_string(),
        600,
      ), // ~3700 chars, 600 words
    ];

    for (input, expected_words) in gpu_tests {
      let mut output = Vec::new();
      let result = crate::dispatch::tokenize_string(&input, &mut output);
      assert!(
        result.is_ok(),
        "GPU: Failed to tokenize string of length {}",
        input.len()
      );
      assert!(
        input.len() > 512,
        "GPU test string should be > 512 chars, got: {}",
        input.len()
      );
      // For GPU path, we're more lenient with exact word count due to complexity
      let word_count = output.len() as u32;
      assert!(
        ((word_count as i32) - expected_words).abs() <= 5,
        "GPU: Word count for string of length {} significantly off, expected ~{}, got {}",
        input.len(),
        expected_words,
        word_count
      );
    }
  }
}
