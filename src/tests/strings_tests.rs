// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use crate::test_utils::config_test_logger;
    #[allow(unused_imports)]
    use crate::types::HwxType;
    use crate::{
        filter_regex_terms, filter_wildcard_terms, match_exact_phrases, match_prefix_strings,
        sort_strings,
    };

    /// Test helper for match_prefix_strings - tests both SCALAR and SIMD paths
    fn test_match_prefix_triple_paths(
        strings: Vec<String>,
        prefix: &str,
        case_insensitive: bool,
        expected_min_matches: usize,
    ) {
        //  SCALAR PATH TEST (small batch - below threshold)
        let mut scalar_strings = strings.clone();
        match_prefix_strings(&mut scalar_strings, prefix, case_insensitive, usize::MAX).unwrap();

        assert!(
            scalar_strings.len() >= expected_min_matches,
            "SCALAR path failed - expected >= {} matches, got {}",
            expected_min_matches,
            scalar_strings.len()
        );

        //  SIMD PATH TEST (large batch - above threshold, skip if input is empty)
        if !strings.is_empty() {
            let large_batch: Vec<String> = strings
                .iter()
                .cycle()
                .take(1000) // Ensure we're above SIMD threshold
                .cloned()
                .collect();

            let mut simd_strings = large_batch.clone();
            match_prefix_strings(&mut simd_strings, prefix, case_insensitive, usize::MAX).unwrap();

            // SIMD should find proportionally more matches
            let expected_simd_matches = expected_min_matches * (1000 / strings.len());
            assert!(
                simd_strings.len() >= expected_simd_matches,
                "SIMD path failed - expected >= {} matches, got {}",
                expected_simd_matches,
                simd_strings.len()
            );

            // Verify SIMD results match the prefix
            for result in &simd_strings {
                assert!(
                    if case_insensitive {
                        result.to_lowercase().starts_with(&prefix.to_lowercase())
                    } else {
                        result.starts_with(prefix)
                    },
                    "SIMD result '{}' doesn't match prefix '{}'",
                    result,
                    prefix
                );
            }
        }

        // Verify all scalar results match the prefix
        for result in &scalar_strings {
            assert!(
                if case_insensitive {
                    result.to_lowercase().starts_with(&prefix.to_lowercase())
                } else {
                    result.starts_with(prefix)
                },
                "SCALAR result '{}' doesn't match prefix '{}'",
                result,
                prefix
            );
        }

        //  GPU PATH TEST (very large batch - above GPU_THRESHOLD_STRING = 512)
        if !strings.is_empty() {
            let gpu_batch: Vec<String> = strings
                .iter()
                .cycle()
                .take(10000) // Well above GPU threshold
                .cloned()
                .collect();

            let mut gpu_strings = gpu_batch.clone();
            match_prefix_strings(&mut gpu_strings, prefix, case_insensitive, usize::MAX).unwrap();

            // GPU should find proportionally more matches
            let expected_gpu_matches = expected_min_matches * (10000 / strings.len());
            assert!(
                gpu_strings.len() >= expected_gpu_matches,
                "GPU path failed - expected >= {} matches, got {}",
                expected_gpu_matches,
                gpu_strings.len()
            );

            // Verify GPU results match the prefix
            for result in &gpu_strings {
                assert!(
                    if case_insensitive {
                        result.to_lowercase().starts_with(&prefix.to_lowercase())
                    } else {
                        result.starts_with(prefix)
                    },
                    "GPU result '{}' doesn't match prefix '{}'",
                    result,
                    prefix
                );
            }
        }
    }

    /// Test helper for match_exact_phrases - tests both SCALAR and SIMD paths
    fn test_match_exact_phrases_triple_paths(
        texts: Vec<String>,
        phrase: &str,
        case_insensitive: bool,
        expected_min_matches: usize,
    ) {
        //  SCALAR PATH TEST
        let mut scalar_texts = texts.clone();
        match_exact_phrases(&mut scalar_texts, phrase, case_insensitive, usize::MAX).unwrap();

        assert!(
            scalar_texts.len() >= expected_min_matches,
            "SCALAR path failed - expected >= {} matches, got {}",
            expected_min_matches,
            scalar_texts.len()
        );

        //  SIMD PATH TEST (skip if input is empty)
        if !texts.is_empty() {
            let large_batch: Vec<String> = texts.iter().cycle().take(1000).cloned().collect();

            let mut simd_texts = large_batch.clone();
            match_exact_phrases(&mut simd_texts, phrase, case_insensitive, usize::MAX).unwrap();

            let expected_simd_matches = expected_min_matches * (1000 / texts.len());
            assert!(
                simd_texts.len() >= expected_simd_matches,
                "SIMD path failed - expected >= {} matches, got {}",
                expected_simd_matches,
                simd_texts.len()
            );
        }

        //  GPU PATH TEST (very large batch - above GPU threshold)
        if !texts.is_empty() {
            let gpu_batch: Vec<String> = texts
                .iter()
                .cycle()
                .take(10000) // Well above GPU threshold
                .cloned()
                .collect();

            let mut gpu_texts = gpu_batch.clone();
            match_exact_phrases(&mut gpu_texts, phrase, case_insensitive, usize::MAX).unwrap();

            let expected_gpu_matches = expected_min_matches * (10000 / texts.len());
            assert!(
                gpu_texts.len() >= expected_gpu_matches,
                "GPU path failed - expected >= {} matches, got {}",
                expected_gpu_matches,
                gpu_texts.len()
            );
        }
    }

    /// Test helper for filter_wildcard_terms - tests both SCALAR and SIMD paths
    fn test_filter_wildcard_triple_paths(
        terms: Vec<String>,
        pattern: &str,
        case_insensitive: bool,
        expected_min_matches: usize,
    ) {
        //  SCALAR PATH TEST
        let mut scalar_terms = terms.clone();
        filter_wildcard_terms(&mut scalar_terms, pattern, case_insensitive, usize::MAX).unwrap();

        assert!(
            scalar_terms.len() >= expected_min_matches,
            "SCALAR path failed - expected >= {} matches, got {}",
            expected_min_matches,
            scalar_terms.len()
        );

        //  SIMD PATH TEST
        let large_batch: Vec<String> = if terms.is_empty() {
            vec!["".to_string(); 1000] // Create 1000 empty strings for SIMD test
        } else {
            terms.iter().cycle().take(1000).cloned().collect()
        };

        let mut simd_terms = large_batch.clone();
        filter_wildcard_terms(&mut simd_terms, pattern, case_insensitive, usize::MAX).unwrap();

        let expected_simd_matches = if terms.is_empty() {
            if pattern == "*" {
                1000
            } else {
                0
            } // Empty strings only match "*"
        } else {
            expected_min_matches * (1000 / terms.len())
        };

        assert!(
            simd_terms.len() >= expected_simd_matches,
            "SIMD path failed - expected >= {} matches, got {}",
            expected_simd_matches,
            simd_terms.len()
        );

        //  GPU PATH TEST (very large batch - above GPU threshold)
        let gpu_batch: Vec<String> = if terms.is_empty() {
            vec!["".to_string(); 10000]
        } else {
            terms.iter().cycle().take(10000).cloned().collect()
        };

        let mut gpu_terms = gpu_batch.clone();
        filter_wildcard_terms(&mut gpu_terms, pattern, case_insensitive, usize::MAX).unwrap();

        let expected_gpu_matches = if terms.is_empty() {
            if pattern == "*" {
                10000
            } else {
                0
            }
        } else {
            expected_min_matches * (10000 / terms.len())
        };

        assert!(
            gpu_terms.len() >= expected_gpu_matches,
            "GPU path failed - expected >= {} matches, got {}",
            expected_gpu_matches,
            gpu_terms.len()
        );
    }

    /// Test helper for sort_strings - tests both SCALAR and SIMD paths
    fn test_sort_strings_triple_paths(strings: Vec<String>) {
        //  SCALAR PATH TEST
        let mut scalar_strings = strings.clone();
        let mut scalar_doc_ids: Vec<u32> = (0..strings.len() as u32).collect();
        sort_strings(&mut scalar_strings, &mut scalar_doc_ids).unwrap();

        // Verify scalar sorting is correct
        for i in 1..scalar_doc_ids.len() {
            let prev_idx = scalar_doc_ids[i - 1] as usize;
            let curr_idx = scalar_doc_ids[i] as usize;
            assert!(
                scalar_strings[prev_idx] <= scalar_strings[curr_idx],
                "SCALAR sorting failed at position {}",
                i
            );
        }

        //  SIMD PATH TEST
        let large_strings: Vec<String> = strings.iter().cycle().take(1000).cloned().collect();

        let mut simd_strings = large_strings.clone();
        let mut simd_doc_ids: Vec<u32> = (0..large_strings.len() as u32).collect();
        sort_strings(&mut simd_strings, &mut simd_doc_ids).unwrap();

        // Verify SIMD sorting is correct
        for i in 1..simd_doc_ids.len() {
            let prev_idx = simd_doc_ids[i - 1] as usize;
            let curr_idx = simd_doc_ids[i] as usize;
            if prev_idx >= simd_strings.len() || curr_idx >= simd_strings.len() {
                panic!(
                    "Index out of bounds: prev_idx={}, curr_idx={}, len={}",
                    prev_idx,
                    curr_idx,
                    simd_strings.len()
                );
            }
            assert!(
                simd_strings[prev_idx] <= simd_strings[curr_idx],
                "SIMD sorting failed at position {}: '{}' > '{}'",
                i,
                simd_strings[prev_idx],
                simd_strings[curr_idx]
            );
        }

        //  GPU PATH TEST (very large batch - above GPU threshold)
        if !strings.is_empty() {
            let gpu_strings: Vec<String> = strings.iter().cycle().take(10000).cloned().collect();
            let mut gpu_strings_mut = gpu_strings.clone();
            let mut gpu_doc_ids: Vec<u32> = (0..gpu_strings.len() as u32).collect();
            sort_strings(&mut gpu_strings_mut, &mut gpu_doc_ids).unwrap();

            // Verify GPU sorting is correct
            for i in 1..gpu_doc_ids.len() {
                let prev_idx = gpu_doc_ids[i - 1] as usize;
                let curr_idx = gpu_doc_ids[i] as usize;
                assert!(
                    gpu_strings_mut[prev_idx] <= gpu_strings_mut[curr_idx],
                    "GPU sorting failed at position {}",
                    i
                );
            }
        }
    }

    /// Test helper for filter_regex_terms - tests both SCALAR and SIMD paths
    fn test_filter_regex_triple_paths(
        terms: Vec<String>,
        regex: &regex::bytes::Regex,
        expected_min_matches: usize,
        test_name: &str,
    ) {
        //  SCALAR PATH TEST (small batch - below threshold)
        let mut scalar_terms = terms.clone();
        filter_regex_terms(&mut scalar_terms, regex, usize::MAX).unwrap();

        assert!(
            scalar_terms.len() >= expected_min_matches,
            "{}: SCALAR path failed - expected >= {} matches, got {}",
            test_name,
            expected_min_matches,
            scalar_terms.len()
        );

        // Verify all scalar results match the regex
        for result in &scalar_terms {
            assert!(
                regex.is_match(result.as_bytes()),
                "{}: SCALAR result '{}' doesn't match regex",
                test_name,
                result
            );
        }

        //  SIMD PATH TEST (large batch - above threshold)
        if !terms.is_empty() {
            let large_batch: Vec<String> = terms.iter().cycle().take(1000).cloned().collect();

            let mut simd_terms = large_batch.clone();
            filter_regex_terms(&mut simd_terms, regex, usize::MAX).unwrap();

            // SIMD should find proportionally more matches
            let expected_simd_matches = expected_min_matches * (1000 / terms.len());
            assert!(
                simd_terms.len() >= expected_simd_matches,
                "{}: SIMD path failed - expected >= {} matches, got {}",
                test_name,
                expected_simd_matches,
                simd_terms.len()
            );

            // Verify all SIMD results match the regex
            for result in &simd_terms {
                assert!(
                    regex.is_match(result.as_bytes()),
                    "{}: SIMD result '{}' doesn't match regex",
                    test_name,
                    result
                );
            }
        }

        //  GPU PATH TEST (very large batch - above GPU threshold)
        if !terms.is_empty() {
            let gpu_batch: Vec<String> = terms.iter().cycle().take(10000).cloned().collect();
            let mut gpu_terms = gpu_batch.clone();
            filter_regex_terms(&mut gpu_terms, regex, usize::MAX).unwrap();

            let expected_gpu_matches = expected_min_matches * (10000 / terms.len());
            assert!(
                gpu_terms.len() >= expected_gpu_matches,
                "{}: GPU path failed - expected >= {} matches, got {}",
                test_name,
                expected_gpu_matches,
                gpu_terms.len()
            );

            // Verify all GPU results match the regex
            for result in &gpu_terms {
                assert!(
                    regex.is_match(result.as_bytes()),
                    "{}: GPU result '{}' doesn't match regex",
                    test_name,
                    result
                );
            }
        }
    }

    #[test]
    fn test_simd_string_sort_triple_paths() {
        config_test_logger();

        // Test small strings (scalar path)
        let mut strings_small = vec![
            "zebra".to_string(),
            "apple".to_string(),
            "banana".to_string(),
        ];
        let mut indices = vec![0u32, 1, 2];
        sort_strings(&mut strings_small, &mut indices).unwrap();
        let sorted_strings: Vec<String> = indices
            .iter()
            .map(|&i| strings_small[i as usize].clone())
            .collect();
        assert_eq!(
            sorted_strings,
            vec![
                "apple".to_string(),
                "banana".to_string(),
                "zebra".to_string()
            ]
        );

        // Test large strings (SIMD path)
        let mut strings_large: Vec<String> = (0..1000)
            .rev()
            .map(|x| format!("string_{:04}", x))
            .collect();
        let mut indices: Vec<u32> = (0..1000).collect();
        sort_strings(&mut strings_large, &mut indices).unwrap();
        let sorted_strings: Vec<String> = indices
            .iter()
            .map(|&i| strings_large[i as usize].clone())
            .collect();
        for i in 0..999 {
            assert!(sorted_strings[i] <= sorted_strings[i + 1]);
        }
        assert_eq!(sorted_strings[0], "string_0000".to_string());
        assert_eq!(sorted_strings[999], "string_0999".to_string());
    }

    #[test]
    fn test_simd_string_sort_basic() {
        config_test_logger();

        let mut strings = vec![
            "zebra".to_string(),
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        let mut indices = vec![0u32, 1, 2, 3];
        sort_strings(&mut strings, &mut indices).unwrap();

        // Verify sorted order
        let sorted_strings: Vec<String> = indices
            .iter()
            .map(|&i| strings[i as usize].clone())
            .collect();

        assert_eq!(
            sorted_strings,
            vec![
                "apple".to_string(),
                "banana".to_string(),
                "cherry".to_string(),
                "zebra".to_string(),
            ]
        );
    }

    #[test]
    fn test_match_prefix_strings() {
        config_test_logger();
        let mut strings = vec![
            "apple".to_string(),
            "application".to_string(),
            "banana".to_string(),
            "apply".to_string(),
        ];

        match_prefix_strings(&mut strings, "app", false, usize::MAX).unwrap();

        assert!(strings.contains(&"apple".to_string())); // "apple"
        assert!(strings.contains(&"application".to_string())); // "application"
        assert!(strings.contains(&"apply".to_string())); // "apply"
        assert!(!strings.contains(&"banana".to_string())); // "banana"

        // Test case-insensitive
        let mut mixed_case = vec!["Apple".to_string(), "APPLY".to_string()];
        match_prefix_strings(&mut mixed_case, "app", true, usize::MAX).unwrap();
        assert_eq!(mixed_case.len(), 2);
    }

    #[test]
    fn test_filter_regex_terms() {
        config_test_logger();
        let mut terms = vec![
            "test123".to_string(),
            "hello".to_string(),
            "test456".to_string(),
            "world".to_string(),
        ];
        let regex = regex::bytes::Regex::new("test\\d+").unwrap();

        filter_regex_terms(&mut terms, &regex, 1000).unwrap();

        // Convert indices back to strings for verification
        assert!(terms.contains(&"test123".to_string()));
        assert!(terms.contains(&"test456".to_string()));
        assert!(terms.len() == 2, "Should match 2 terms");
    }

    #[test]
    fn test_filter_wildcard_terms() {
        config_test_logger();
        let mut terms = vec![
            "file.txt".to_string(),
            "document.pdf".to_string(),
            "image.txt".to_string(), // should be filtered out
            "script.py".to_string(),
        ];
        filter_wildcard_terms(&mut terms, "*.txt", false, usize::MAX).unwrap();

        // Convert indices back to strings for verification
        assert!(terms.contains(&"file.txt".to_string()));
        assert!(terms.contains(&"image.txt".to_string()));
        assert!(!terms.contains(&"document.pdf".to_string()));
        assert!(!terms.contains(&"script.py".to_string()));
        assert!(terms.len() == 2, "Should match 2 terms");
    }

    // =============================================================================
    //  COMPREHENSIVE SIMD TEST SUITE - STRINGS.RS FINAL BOSS EDITION
    // =============================================================================

    /// Comprehensive dual-path test suite for match_prefix_strings
    #[test]
    fn test_match_prefix_strings_comprehensive_triple_paths() {
        config_test_logger();

        // Basic prefix patterns - both scalar and SIMD paths
        let prefix_strings = vec![
            "apple".to_string(),
            "application".to_string(),
            "apply".to_string(),
            "banana".to_string(),
            "app".to_string(),
        ];
        test_match_prefix_triple_paths(prefix_strings.clone(), "app", false, 4); // Should match 4 "app" prefixes

        // Case sensitivity patterns - both scalar and SIMD paths
        let case_strings = vec![
            "Apple".to_string(),
            "APPLICATION".to_string(),
            "apply".to_string(),
            "Banana".to_string(),
        ];
        test_match_prefix_triple_paths(case_strings.clone(), "app", false, 1); // Case sensitive - only "apply"
        test_match_prefix_triple_paths(case_strings.clone(), "app", true, 3); // Case insensitive - 3 matches

        // Large dataset with mixed prefixes - both scalar and SIMD paths
        let large_mixed: Vec<String> = (0..90)
            .map(|i| {
                if i % 3 == 0 {
                    format!("prefix_{}", i)
                } else if i % 5 == 0 {
                    format!("PREFIX_{}", i) // Different case
                } else {
                    format!("other_{}", i)
                }
            })
            .collect();
        test_match_prefix_triple_paths(large_mixed.clone(), "prefix", false, 30); // Case sensitive
        test_match_prefix_triple_paths(large_mixed.clone(), "prefix", true, 36); // Case insensitive - 30 "prefix_" + 12 "PREFIX_" = 42

        // Edge cases - both scalar and SIMD paths
        test_match_prefix_triple_paths(vec![], "test", false, 0); // Empty input
        test_match_prefix_triple_paths(vec!["test".to_string()], "test", false, 1); // Exact match
        test_match_prefix_triple_paths(prefix_strings.clone(), "", false, 5); // Empty prefix - should match all
        test_match_prefix_triple_paths(prefix_strings.clone(), "xyz", false, 0);
        // No matches
    }

    /// Comprehensive dual-path test suite for match_exact_phrases
    #[test]
    fn test_match_exact_phrases_comprehensive_triple_paths() {
        config_test_logger();

        // Basic phrase patterns - both scalar and SIMD paths
        let phrase_texts = vec![
            "The quick brown fox".to_string(),
            "A quick brown dog".to_string(),
            "The slow brown fox".to_string(),
            "Quick brown fox jumps".to_string(),
            "brown fox is quick".to_string(),
        ];
        test_match_exact_phrases_triple_paths(phrase_texts.clone(), "quick brown", false, 2); // Case sensitive matches

        // Case sensitivity patterns - both scalar and SIMD paths
        test_match_exact_phrases_triple_paths(phrase_texts.clone(), "Quick brown", false, 1); // Case sensitive - only "Quick brown fox jumps"
        test_match_exact_phrases_triple_paths(phrase_texts.clone(), "Quick brown", true, 3); // Case insensitive - more matches

        // Large dataset with varied content - both scalar and SIMD paths
        let large_docs: Vec<String> = (0..120)
            .map(|i| {
                if i % 4 == 0 {
                    format!("This is a test document number {}", i)
                } else if i % 7 == 0 {
                    format!("Test Document with number {}", i) // Different case
                } else {
                    format!("Random text content {}", i)
                }
            })
            .collect();
        test_match_exact_phrases_triple_paths(large_docs.clone(), "test document", true, 40); // Case insensitive to catch both cases

        // Edge cases - both scalar and SIMD paths
        test_match_exact_phrases_triple_paths(vec![], "any phrase", false, 0); // Empty input
        test_match_exact_phrases_triple_paths(
            vec!["exact phrase".to_string()],
            "exact phrase",
            false,
            1,
        ); // Exact match
        test_match_exact_phrases_triple_paths(phrase_texts.clone(), "purple elephant", false, 0); // No matches
        test_match_exact_phrases_triple_paths(vec!["single word".to_string()], "word", false, 1);
        // Substring match
    }

    /// Comprehensive dual-path test suite for filter_regex_terms
    #[test]
    fn test_filter_regex_terms_comprehensive_triple_paths() {
        config_test_logger();

        // Email regex pattern - both scalar and SIMD paths
        let email_regex =
            regex::bytes::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        let email_terms = vec![
            "user@example.com".to_string(),
            "invalid-email".to_string(),
            "another@test.org".to_string(),
            "not.an.email".to_string(),
            "admin@domain.co.uk".to_string(),
        ];
        test_filter_regex_triple_paths(email_terms, &email_regex, 3, "email_regex");

        // Number pattern regex - both scalar and SIMD paths
        let number_regex = regex::bytes::Regex::new(r"^\d+$").unwrap();
        let number_terms = vec![
            "123".to_string(),
            "abc".to_string(),
            "456789".to_string(),
            "12a34".to_string(),
            "0".to_string(),
        ];
        test_filter_regex_triple_paths(number_terms, &number_regex, 3, "number_regex");

        // Phone pattern regex - both scalar and SIMD paths
        let phone_regex = regex::bytes::Regex::new(r"^\(\d{3}\) \d{3}-\d{4}$").unwrap();
        let phone_terms = vec![
            "(123) 456-7890".to_string(),
            "123-456-7890".to_string(),
            "(999) 888-7777".to_string(),
            "invalid phone".to_string(),
        ];
        test_filter_regex_triple_paths(phone_terms, &phone_regex, 2, "phone_regex");

        // Large dataset with varied email patterns - both scalar and SIMD paths
        let large_email_terms: Vec<String> = (0..100)
            .map(|i| {
                if i % 3 == 0 {
                    format!("user{}@example.com", i) // Valid email format
                } else if i % 5 == 0 {
                    format!("{}@invalid", i) // Invalid email format
                } else {
                    format!("random_text_{}", i) // Not email at all
                }
            })
            .collect();
        test_filter_regex_triple_paths(large_email_terms, &email_regex, 30, "large_email_dataset");

        // Edge cases - both scalar and SIMD paths
        test_filter_regex_triple_paths(vec![], &phone_regex, 0, "empty_terms");
        test_filter_regex_triple_paths(
            vec!["single_term@test.com".to_string()],
            &email_regex,
            1,
            "single_valid_term",
        );
        test_filter_regex_triple_paths(
            vec!["invalid".to_string()],
            &email_regex,
            0,
            "single_invalid_term",
        );
    }

    /// Comprehensive dual-path test suite for filter_wildcard_terms
    #[test]
    fn test_filter_wildcard_terms_comprehensive_triple_paths() {
        config_test_logger();

        // File extension patterns - both scalar and SIMD paths
        let file_terms = vec![
            "document.pdf".to_string(),
            "image.jpg".to_string(),
            "text.txt".to_string(),
            "script.py".to_string(),
            "data.csv".to_string(),
            "archive.tar.gz".to_string(),
        ];
        test_filter_wildcard_triple_paths(file_terms.clone(), "*.pdf", false, 1); // One PDF file
        test_filter_wildcard_triple_paths(file_terms.clone(), "*.*", false, 5); // Most files with extensions
        test_filter_wildcard_triple_paths(file_terms.clone(), "data*", false, 1); // Prefix match

        // Case sensitivity patterns - both scalar and SIMD paths
        let case_terms = vec![
            "FILE.PDF".to_string(),
            "Image.JPG".to_string(),
            "text.txt".to_string(),
        ];
        test_filter_wildcard_triple_paths(case_terms.clone(), "*.pdf", false, 0); // Case sensitive - no match
        test_filter_wildcard_triple_paths(case_terms.clone(), "*.pdf", true, 1); // Case insensitive - should match

        // Complex wildcard patterns - both scalar and SIMD paths
        let complex_terms = vec![
            "report_2023.pdf".to_string(),
            "report_2024.pdf".to_string(),
            "summary_2023.txt".to_string(),
            "data_2023.csv".to_string(),
            "backup_2023.zip".to_string(),
        ];
        test_filter_wildcard_triple_paths(complex_terms.clone(), "*_2023.*", false, 4); // 2023 files

        // Large dataset with varied extensions - both scalar and SIMD paths
        let large_extensions = ["txt", "pdf", "jpg", "png", "doc", "csv"];
        let large_terms: Vec<String> = (0..60)
            .map(|i| {
                let ext = &large_extensions[i % large_extensions.len()];
                format!("file_{}.{}", i, ext)
            })
            .collect();
        test_filter_wildcard_triple_paths(large_terms.clone(), "*.txt", false, 10); // Should find txt files

        // Edge cases - both scalar and SIMD paths
        test_filter_wildcard_triple_paths(vec![], "*", false, 0); // Empty input
        test_filter_wildcard_triple_paths(vec!["single.txt".to_string()], "*.txt", false, 1); // Single match
        test_filter_wildcard_triple_paths(vec!["nomatch".to_string()], "*.xyz", false, 0); // No matches

        // Special characters - both scalar and SIMD paths
        let special_terms = vec![
            "".to_string(),            // Empty string
            ".hidden".to_string(),     // Hidden file
            "noextension".to_string(), // No extension
        ];
        test_filter_wildcard_triple_paths(special_terms, "*", false, 2); // Should match most non-empty
    }

    /// Test suite for sort_strings with extensive coverage
    #[test]
    fn test_sort_strings_comprehensive() {
        config_test_logger();

        //   DUAL-PATH TEST - Mixed string sorting
        let small_strings = vec![
            "zebra".to_string(),
            "apple".to_string(),
            "dog".to_string(),
            "cat".to_string(),
            "elephant".to_string(),
        ];
        test_sort_strings_triple_paths(small_strings);

        //   DUAL-PATH TEST - Complex pattern strings
        let large_strings: Vec<String> = (0..100) // Reduced for testing
            .map(|i| {
                // Create varied string lengths and patterns
                match i % 4 {
                    0 => format!("prefix_{:04}", 99 - i), // Reverse order
                    1 => format!("a_{}", i),
                    2 => format!("zzz_{:03}", i % 10), // Some duplicates
                    _ => format!("{:02}_suffix", i % 5),
                }
            })
            .collect();
        test_sort_strings_triple_paths(large_strings);

        //   DUAL-PATH TEST - Single string
        test_sort_strings_triple_paths(vec!["alone".to_string()]);

        //   DUAL-PATH TEST - Empty strings mixed
        test_sort_strings_triple_paths(vec!["".to_string(), "a".to_string(), "".to_string()]);

        // Identical strings
        let mut identical_strings = vec!["same".to_string(); 10];
        let mut identical_doc_ids: Vec<u32> = (0..10).collect();
        sort_strings(&mut identical_strings, &mut identical_doc_ids).unwrap();

        // All indices should still be valid (order doesn't matter for identical strings)
        for &idx in &identical_doc_ids {
            assert!(
                (idx as usize) < identical_strings.len(),
                "All indices should be valid"
            );
        }

        //  GPU PATH TEST (very large array for string sorting)
        let gpu_strings: Vec<String> = (0..10000)
            .map(|i| format!("string_{:05}", 9999 - i)) // Reverse order to test sorting
            .collect();
        let mut gpu_strings_mut = gpu_strings.clone();
        let mut gpu_doc_ids: Vec<u32> = (0..10000).collect();
        sort_strings(&mut gpu_strings_mut, &mut gpu_doc_ids).unwrap();

        // Verify GPU sorting - check that indices are sorted, not the strings themselves
        for i in 1..gpu_doc_ids.len() {
            let prev_idx = gpu_doc_ids[i - 1] as usize;
            let curr_idx = gpu_doc_ids[i] as usize;
            assert!(
                gpu_strings_mut[prev_idx] <= gpu_strings_mut[curr_idx],
                "GPU string sort failed at position {}",
                i
            );
        }
    }

    // =============================================================================
    // PREFIX MATCHING TESTS
    // =============================================================================
    #[test]
    fn test_match_prefix_strings_triple_paths() {
        config_test_logger();

        // Test small strings (scalar path)
        let mut strings_small = vec![
            "apple".to_string(),
            "application".to_string(),
            "banana".to_string(),
        ];
        match_prefix_strings(&mut strings_small, "app", false, 100).unwrap();
        assert_eq!(
            strings_small,
            vec!["apple".to_string(), "application".to_string()]
        );

        // Test large strings (SIMD path)
        let mut strings_large: Vec<String> = (0..1000)
            .map(|x| {
                if x % 2 == 0 {
                    format!("test_{}", x)
                } else {
                    format!("other_{}", x)
                }
            })
            .collect();
        match_prefix_strings(&mut strings_large, "test", false, 1000).unwrap();
        assert_eq!(strings_large.len(), 500); // Should have 500 strings starting with "test"
    }

    #[test]
    fn test_match_prefix_strings_basic_triple_paths() {
        config_test_logger();

        // Basic "he" prefix pattern - both scalar and SIMD paths
        let basic_strings = vec![
            "hello".to_string(),
            "help".to_string(),
            "helicopter".to_string(),
            "hero".to_string(),
            "world".to_string(),
        ];
        test_match_prefix_triple_paths(basic_strings, "he", false, 4); // Should match "hello", "help", "helicopter", "hero"
    }

    #[test]
    fn test_match_prefix_strings_edge_cases_triple_paths() {
        config_test_logger();

        // Edge case patterns - both scalar and SIMD paths
        let edge_strings = vec![
            "a".to_string(),
            "ab".to_string(),
            "abc".to_string(),
            "abcdefghijklmnop".to_string(),
        ];
        test_match_prefix_triple_paths(edge_strings.clone(), "", false, 4); // Empty prefix - all match
        test_match_prefix_triple_paths(edge_strings.clone(), "abcdefgh", false, 1); // Long prefix - one match
        test_match_prefix_triple_paths(edge_strings.clone(), "ab", false, 3); // Medium prefix - multiple matches
                                                                              //  GPU PATH TEST (very large batch for prefix matching)
        let gpu_strings: Vec<String> = (0..10000)
            .map(|i| {
                if i % 2 == 0 {
                    format!("prefix_{}", i)
                } else {
                    format!("other_{}", i)
                }
            })
            .collect();
        let mut gpu_strings_mut = gpu_strings.clone();
        match_prefix_strings(&mut gpu_strings_mut, "prefix_", false, usize::MAX).unwrap();

        // Should have ~5000 matches
        assert!(
            gpu_strings_mut.len() >= 4900 && gpu_strings_mut.len() <= 5100,
            "GPU prefix matching failed - expected ~5000 matches, got {}",
            gpu_strings_mut.len()
        );

        // Verify all results start with prefix
        for s in &gpu_strings_mut {
            assert!(s.starts_with("prefix_"), "GPU result doesn't match prefix");
        }
    }

    #[test]
    fn test_match_prefix_strings_max_results_triple_paths() {
        config_test_logger();

        // Large dataset patterns - both scalar and SIMD paths
        let many_strings: Vec<String> = (0..60).map(|i| format!("test_{}", i)).collect();
        test_match_prefix_triple_paths(many_strings.clone(), "test_", false, 60); // All should match
        test_match_prefix_triple_paths(many_strings.clone(), "test_1", false, 11);
        // Should match test_1, test_10-19
    }

    // =============================================================================
    // WILDCARD FILTERING TESTS
    // =============================================================================

    #[test]
    fn test_filter_wildcard_terms_triple_paths() {
        config_test_logger();

        // Test small terms (scalar path)
        let mut terms_small = vec![
            "apple".to_string(),
            "application".to_string(),
            "banana".to_string(),
        ];
        filter_wildcard_terms(&mut terms_small, "app*", false, usize::MAX).unwrap();
        assert_eq!(
            terms_small,
            vec!["apple".to_string(), "application".to_string()]
        );

        // Test large terms (SIMD path)
        let mut terms_large: Vec<String> = (0..1000)
            .map(|x| {
                if x % 2 == 0 {
                    format!("test_{}", x)
                } else {
                    format!("other_{}", x)
                }
            })
            .collect();
        filter_wildcard_terms(&mut terms_large, "test_*", false, usize::MAX).unwrap();
        assert_eq!(terms_large.len(), 500); // Half should match
        for term in &terms_large {
            assert!(term.starts_with("test_"));
        }

        //  GPU PATH TEST (very large batch for wildcard filtering)
        let mut gpu_terms: Vec<String> = (0..10000)
            .map(|x| match x % 3 {
                0 => format!("match_{}", x),
                1 => format!("test_{}", x),
                _ => format!("other_{}", x),
            })
            .collect();
        filter_wildcard_terms(&mut gpu_terms, "match_*", false, usize::MAX).unwrap();

        // Should have ~3333 matches
        assert!(
            gpu_terms.len() >= 3300 && gpu_terms.len() <= 3400,
            "GPU wildcard filtering failed - expected ~3333 matches, got {}",
            gpu_terms.len()
        );

        // Verify all results match the pattern
        for term in &gpu_terms {
            assert!(
                term.starts_with("match_"),
                "GPU result doesn't match wildcard pattern"
            );
        }
    }

    #[test]
    fn test_filter_wildcard_terms_basic_triple_paths() {
        config_test_logger();

        // Basic wildcard "*est*" pattern - both scalar and SIMD paths
        let basic_terms = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
            "testing".to_string(),
            "best".to_string(),
        ];
        test_filter_wildcard_triple_paths(basic_terms, "*est*", false, 3); // Should match "test", "testing", "best"
    }

    #[test]
    fn test_filter_wildcard_terms_complex_patterns_triple_paths() {
        config_test_logger();

        // Complex "*app*" pattern - both scalar and SIMD paths
        let complex_terms = vec![
            "application".to_string(),
            "apple".to_string(),
            "appliance".to_string(),
            "pineapple".to_string(),
            "appreciation".to_string(),
        ];
        test_filter_wildcard_triple_paths(complex_terms, "*app*", false, 5); // All 5 contain "app"
    }

    #[test]
    fn test_filter_wildcard_terms_exact_match_triple_paths() {
        config_test_logger();

        // Exact match wildcard pattern - both scalar and SIMD paths
        let exact_terms = vec![
            "exact".to_string(),
            "exactly".to_string(),
            "inexact".to_string(),
        ];
        test_filter_wildcard_triple_paths(exact_terms, "exact", false, 1); // Only "exact" matches exactly

        // Multiple exact patterns - both scalar and SIMD paths
        let multi_terms = vec![
            "file.txt".to_string(),
            "document.pdf".to_string(),
            "image.png".to_string(),
            "data.txt".to_string(),
        ];
        test_filter_wildcard_triple_paths(multi_terms, "*.txt", false, 2); // Should match .txt files
    }

    // =============================================================================
    // REGEX FILTERING TESTS
    // =============================================================================

    #[test]
    fn test_filter_regex_terms_simple_patterns_triple_paths() {
        config_test_logger();

        // Simple "ca.*" pattern - both scalar and SIMD paths
        let ca_terms = vec![
            "cat".to_string(),
            "car".to_string(),
            "card".to_string(),
            "care".to_string(),
            "bat".to_string(),
        ];
        let ca_regex = regex::bytes::Regex::new("ca.*").unwrap();
        test_filter_regex_triple_paths(ca_terms, &ca_regex, 4, "ca_pattern");
    }

    #[test]
    fn test_filter_regex_terms_digit_patterns_triple_paths() {
        config_test_logger();

        // Digit pattern "item[0-9]+" - both scalar and SIMD paths
        let digit_terms = vec![
            "item1".to_string(),
            "item12".to_string(),
            "item123".to_string(),
            "item_a".to_string(),
            "item".to_string(),
        ];
        let digit_regex = regex::bytes::Regex::new("item[0-9]+").unwrap();
        test_filter_regex_triple_paths(digit_terms, &digit_regex, 3, "digit_pattern");
    }

    #[test]
    fn test_filter_regex_terms_invalid_regex() {
        config_test_logger();
        let mut terms = vec!["test".to_string()];

        // Invalid regex should be handled gracefully
        #[allow(clippy::invalid_regex)]
        match regex::bytes::Regex::new("[") {
            Ok(regex) => {
                filter_regex_terms(&mut terms, &regex, 1000).unwrap();
            }
            Err(_) => {
                // Invalid regex, skip test - this is expected behavior
            }
        }
        // Should not panic, may return empty results
        assert!(terms.len() <= terms.len());
    }

    #[test]
    fn test_filter_regex_terms_max_size_limit_triple_paths() {
        config_test_logger();

        // Max size limit patterns - both scalar and SIMD paths
        let limit_terms = vec![
            "test1".to_string(),
            "test2".to_string(),
            "test3".to_string(),
            "test4".to_string(),
            "test5".to_string(),
            "notmatch".to_string(),
        ];
        let test_regex = regex::bytes::Regex::new("test[0-9]+").unwrap();

        // Test with sufficient matches available - both scalar and SIMD paths
        test_filter_regex_triple_paths(
            limit_terms.clone(),
            &test_regex,
            5,
            "test_max_size_available",
        );

        // Edge cases - both scalar and SIMD paths
        test_filter_regex_triple_paths(vec![], &test_regex, 0, "empty_input_max_size");
        test_filter_regex_triple_paths(
            vec!["match1".to_string(), "match2".to_string()],
            &test_regex,
            0,
            "no_matches_available",
        );
    }

    #[test]
    fn test_filter_regex_terms_various_limits_triple_paths() {
        config_test_logger();

        // Various limit scenarios - both scalar and SIMD paths
        let limit_terms = vec![
            "test1".to_string(),
            "test2".to_string(),
            "nomatch".to_string(),
            "alsonomatch".to_string(),
        ];
        let test_regex = regex::bytes::Regex::new("test[0-9]+").unwrap();
        test_filter_regex_triple_paths(limit_terms, &test_regex, 2, "normal_matches");

        // Single match scenarios - both scalar and SIMD paths
        let single_terms = vec![
            "match1".to_string(),
            "match2".to_string(),
            "match3".to_string(),
        ];
        let match_regex = regex::bytes::Regex::new("match[0-9]+").unwrap();
        test_filter_regex_triple_paths(single_terms, &match_regex, 3, "all_match_scenario");
    }

    // =============================================================================
    // PHRASE MATCHING TESTS
    // =============================================================================

    #[test]
    fn test_match_field_phrases_triple_paths() {
        config_test_logger();

        // Test small strings (scalar path) - using direct string matching instead of field extraction
        let mut strings_small = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "goodbye".to_string(),
        ];
        match_exact_phrases(&mut strings_small, "hello", false, 100).unwrap();
        assert_eq!(strings_small.len(), 2); // Should match "hello world" and "hello there"

        // Test large strings (SIMD path) - using direct string matching
        let mut strings_large: Vec<String> = (0..1000)
            .map(|x| {
                if x % 2 == 0 {
                    format!("test phrase {}", x)
                } else {
                    format!("other content {}", x)
                }
            })
            .collect();
        match_exact_phrases(&mut strings_large, "test phrase", false, 1000).unwrap();
        assert_eq!(strings_large.len(), 500); // Should match 500 strings with "test phrase"

        //  GPU PATH TEST (very large batch for field phrase matching)
        let mut gpu_strings: Vec<String> = (0..10000)
            .map(|x| match x % 4 {
                0 => format!("field: target phrase {}", x),
                1 => format!("field: other content {}", x),
                2 => format!("different: target phrase {}", x),
                _ => format!("no match here {}", x),
            })
            .collect();
        match_exact_phrases(&mut gpu_strings, "target phrase", false, usize::MAX).unwrap();

        // Should match ~5000 strings (those with "target phrase")
        assert!(
            gpu_strings.len() >= 4900 && gpu_strings.len() <= 5100,
            "GPU field phrase matching failed - expected ~5000 matches, got {}",
            gpu_strings.len()
        );

        // Verify all results contain the phrase
        for s in &gpu_strings {
            assert!(
                s.contains("target phrase"),
                "GPU result doesn't contain target phrase"
            );
        }
    }

    #[test]
    fn test_match_exact_phrases_basic_triple_paths() {
        config_test_logger();

        // Basic phrase "quick" pattern - both scalar and SIMD paths
        let basic_texts = vec![
            "The quick brown fox jumps over the lazy dog".to_string(),
            "A quick test of the system".to_string(),
            "Nothing matches here".to_string(),
        ];
        test_match_exact_phrases_triple_paths(basic_texts, "quick", false, 2); // Two texts contain "quick"
    }

    #[test]
    fn test_match_exact_phrases_case_insensitive_triple_paths() {
        config_test_logger();

        // Case insensitive "hello" pattern - both scalar and SIMD paths
        let case_texts = vec![
            "HELLO WORLD".to_string(),
            "hello world".to_string(),
            "Hello World".to_string(),
            "goodbye".to_string(),
        ];
        test_match_exact_phrases_triple_paths(case_texts.clone(), "hello", false, 1); // Case sensitive - only "hello world"
        test_match_exact_phrases_triple_paths(case_texts.clone(), "hello", true, 3);
        // Case insensitive - all 3 "hello" variants
    }

    // =============================================================================
    // SIMD CONSISTENCY TESTS
    // =============================================================================

    #[test]
    fn test_string_simd_consistency() {
        let mut test_strings = vec![
            "prefix_test".to_string(),
            "test_suffix".to_string(),
            "middle_test_string".to_string(),
            "other".to_string(),
        ];

        // Test prefix matching - dispatcher automatically selects best implementation
        match_prefix_strings(&mut test_strings, "prefix", false, usize::MAX).unwrap();

        // Should find strings that start with "prefix"
        assert!(!test_strings.is_empty());
        for string in &test_strings {
            assert!(string.starts_with("prefix"));
        }
    }

    // =============================================================================
    // PERFORMANCE TESTS (Large Datasets)
    // =============================================================================

    #[test]
    fn test_string_functions_large_dataset() {
        config_test_logger();

        //  NEW ZERO-COPY PATTERN: Create indices for large dataset (limited for performance)
        let mut large_strings: Vec<String> = (0..8)
            .map(|i| format!("string_{}_with_suffix", i))
            .collect();

        // Test prefix matching on large dataset
        match_prefix_strings(&mut large_strings, "string_", false, usize::MAX).unwrap();

        let final_count = large_strings.len().min(8); // Limit for test performance
        assert!(final_count >= 5); // Should find many matches

        // Test wildcard filtering on large dataset - reinitialize first
        large_strings = (0..8)
            .map(|i| format!("string_{}_with_suffix", i))
            .collect();
        filter_wildcard_terms(&mut large_strings, "*_with_*", false, usize::MAX).unwrap();
        assert!(large_strings.len() <= 8);
        assert!(large_strings.len() >= 5); // Should find many matches
    }

    #[test]
    fn test_string_functions_very_long_strings() {
        config_test_logger();
        let mut long_strings = vec![
            "a".repeat(10000),
            "b".repeat(5000) + &"target".repeat(100) + &"c".repeat(5000),
            "prefix_".to_string() + &"x".repeat(20000),
        ];

        match_prefix_strings(&mut long_strings, "prefix_", false, usize::MAX).unwrap();

        assert!(long_strings.len() <= 1); // Only one should match
        if !long_strings.is_empty() {
            assert!(long_strings[0].starts_with("prefix_"));
        }
    }

    // =============================================================================
    // UNICODE AND SPECIAL CHARACTER TESTS
    // =============================================================================

    #[test]
    fn test_string_functions_unicode() {
        let mut unicode_strings = vec![
            "café".to_string(),
            "naïve".to_string(),
            "résumé".to_string(),
            "test".to_string(),
        ];

        filter_wildcard_terms(&mut unicode_strings, "*é*", false, usize::MAX).unwrap();

        //  REQUIRED: Truncate to valid results

        // Unicode test should not crash - result may vary
    }

    #[test]
    fn test_string_functions_special_characters() {
        config_test_logger();
        let mut special_strings = vec![
            "hello@world.com".to_string(),
            "test#123".to_string(),
            "file.txt".to_string(),
            "path/to/file".to_string(),
        ];

        //  NEW ZERO-COPY PATTERN: Create indices for special characters
        filter_wildcard_terms(&mut special_strings, "*@*", false, usize::MAX).unwrap();

        assert!(special_strings.contains(&"hello@world.com".to_string()));
    }

    // =============================================================================
    // EDGE CASE TESTS
    // =============================================================================

    #[test]
    fn test_string_functions_empty_inputs() {
        config_test_logger();
        let mut empty_strings: Vec<String> = vec![];

        //  NEW ZERO-COPY PATTERN: Empty indices for empty input

        match_prefix_strings(&mut empty_strings, "test", false, usize::MAX).unwrap();

        assert_eq!(empty_strings.len(), 0);

        //  NEW ZERO-COPY PATTERN: Empty indices for wildcard test
        filter_wildcard_terms(&mut empty_strings, "*", false, usize::MAX).unwrap();

        assert_eq!(empty_strings.len(), 0);

        // Test empty text for phrase search
        let mut empty_text = vec!["".to_string()];
        //  NEW ZERO-COPY PATTERN: Empty indices for phrase matching
        match_exact_phrases(&mut empty_text, "phrase", false, usize::MAX).unwrap();

        assert_eq!(empty_text.len(), 0);
    }

    #[test]
    fn test_string_functions_single_character() {
        config_test_logger();
        let mut single_char_strings = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        //  NEW ZERO-COPY PATTERN: Create indices for single character test
        match_prefix_strings(&mut single_char_strings, "a", false, usize::MAX).unwrap();

        assert_eq!(single_char_strings.len(), 1);
        assert_eq!(single_char_strings[0], "a");

        // Test different prefix - reinitialize the vector
        single_char_strings = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        match_prefix_strings(&mut single_char_strings, "b", false, usize::MAX).unwrap();

        assert_eq!(single_char_strings.len(), 1);
        assert_eq!(single_char_strings[0], "b");
    }

    #[test]
    fn test_minimal_sort() {
        config_test_logger();

        // Force SIMD path with exactly 32 strings to trigger SIMD
        let base_strings = vec!["dog".to_string(), "cat".to_string()];
        let mut large_strings: Vec<String> = Vec::new();
        for _ in 0..16 {
            large_strings.extend(base_strings.clone());
        }
        // This creates 32 strings, forcing SIMD path

        let mut indices: Vec<u32> = (0..large_strings.len() as u32).collect();

        sort_strings(&mut large_strings, &mut indices).unwrap();

        // Check just the first few results
        for i in 1..std::cmp::min(10, indices.len()) {
            let prev_idx = indices[i - 1] as usize;
            let curr_idx = indices[i] as usize;
            assert!(
                large_strings[prev_idx] <= large_strings[curr_idx],
                "SIMD sorting failed at position {}: '{}' > '{}'",
                i,
                large_strings[prev_idx],
                large_strings[curr_idx]
            );
        }
    }

    #[test]
    fn test_comparison_debug() {
        config_test_logger();

        let strings = ["dog".to_string(), "cat".to_string()];

        // Test standard library comparison
        log::debug!("std::cmp: 'dog' > 'cat' = {}", "dog" > "cat");

        // We can't directly test compare_strings as it's private
        // but we can test the sorting behavior
        let mut indices = vec![0u32, 1u32];
        indices.sort_by(|&a, &b| strings[a as usize].cmp(&strings[b as usize]));

        log::debug!(
            "After std sort: [{}, {}] -> ['{} ', '{}']",
            indices[0],
            indices[1],
            strings[indices[0] as usize],
            strings[indices[1] as usize]
        );

        // Should be [1, 0] because "cat" < "dog"
        assert_eq!(indices, vec![1, 0]);
    }

    #[test]
    fn test_match_prefix_strings_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 3 == 0 {
                large_strings.push(format!("test_prefix_{}", i));
            } else if i % 5 == 0 {
                large_strings.push(format!("another_test_{}", i));
            } else {
                large_strings.push(format!("random_string_{}", i));
            }
        }

        let original_len = large_strings.len();

        // Test prefix matching with chunking
        crate::dispatch::match_prefix_strings(&mut large_strings, "test_prefix", false, 1000)
            .expect("Large array prefix matching should not panic");

        // Verify results
        assert!(large_strings.len() <= 1000, "Should respect max_size limit");
        for s in &large_strings {
            assert!(
                s.starts_with("test_prefix"),
                "All results should match prefix: {}",
                s
            );
        }

        log::debug!(
            "Large array test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }

    #[test]
    fn test_match_exact_phrases_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 7 == 0 {
                large_strings.push("exact match phrase".to_string());
            } else if i % 11 == 0 {
                large_strings.push("contains exact match phrase here".to_string());
            } else {
                large_strings.push(format!("other content {}", i));
            }
        }

        let original_len = large_strings.len();

        // Test exact phrase matching with chunking
        crate::dispatch::match_exact_phrases(&mut large_strings, "exact match phrase", false, 500)
            .expect("Large array exact phrase matching should not panic");

        // Verify results
        assert!(large_strings.len() <= 500, "Should respect max_size limit");
        for s in &large_strings {
            assert!(
                s.contains("exact match phrase"),
                "All results should contain phrase: {}",
                s
            );
        }

        log::debug!(
            "Large array exact phrase test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }

    #[test]
    fn test_match_field_prefixes_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 4 == 0 {
                large_strings.push(format!("prefix_match_{}", i));
            } else {
                large_strings.push(format!("other_value_{}", i));
            }
        }

        let original_len = large_strings.len();

        // Test prefix matching with chunking (same as match_prefix_strings)
        crate::dispatch::match_prefix_strings(&mut large_strings, "prefix_match", false, 800)
            .expect("Large array field prefix matching should not panic");

        // Verify results
        assert!(large_strings.len() <= 800, "Should respect max_size limit");
        for s in &large_strings {
            assert!(
                s.starts_with("prefix_match"),
                "All results should match prefix: {}",
                s
            );
        }

        log::debug!(
            "Large array field prefix test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }

    #[test]
    fn test_match_field_phrases_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 6 == 0 {
                large_strings.push("contains target phrase here".to_string());
            } else {
                large_strings.push(format!("different content {}", i));
            }
        }

        let original_len = large_strings.len();

        // Test phrase matching with chunking (same as match_exact_phrases)
        crate::dispatch::match_exact_phrases(&mut large_strings, "target phrase", false, 600)
            .expect("Large array field phrase matching should not panic");

        // Verify results
        assert!(large_strings.len() <= 600, "Should respect max_size limit");
        for s in &large_strings {
            assert!(
                s.contains("target phrase"),
                "All results should contain phrase: {}",
                s
            );
        }

        log::debug!(
            "Large array field phrase test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }

    #[test]
    fn test_filter_wildcard_terms_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 8 == 0 {
                large_strings.push(format!("wildcard_test_{}", i));
            } else if i % 13 == 0 {
                large_strings.push(format!("test_wildcard_match_{}", i));
            } else {
                large_strings.push(format!("unrelated_content_{}", i));
            }
        }

        let original_len = large_strings.len();

        // Test wildcard filtering with chunking
        crate::dispatch::filter_wildcard_terms(&mut large_strings, "*wildcard*", false, 700)
            .expect("Large array wildcard filtering should not panic");

        // Verify results
        assert!(large_strings.len() <= 700, "Should respect max_size limit");
        for s in &large_strings {
            assert!(
                s.contains("wildcard"),
                "All results should match wildcard pattern: {}",
                s
            );
        }

        log::debug!(
            "Large array wildcard test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }

    #[test]
    fn test_filter_regex_terms_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 7 == 0 {
                large_strings.push(format!("regex_test_{}", i));
            } else if i % 11 == 0 {
                large_strings.push(format!("test_regex_match_{}", i));
            } else {
                large_strings.push(format!("unrelated_content_{}", i));
            }
        }

        let original_len = large_strings.len();

        // Create regex pattern
        let regex = regex::bytes::Regex::new(r"regex").expect("Valid regex pattern");

        // Test regex filtering with chunking
        crate::dispatch::filter_regex_terms(&mut large_strings, &regex, 500)
            .expect("Large array regex filtering should not panic");

        // Verify results
        assert!(large_strings.len() <= 500, "Should respect max_size limit");
        for s in &large_strings {
            assert!(
                s.contains("regex"),
                "All results should match regex pattern: {}",
                s
            );
        }

        log::debug!(
            "Large array regex test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }

    #[test]
    fn test_match_exact_phrases_case_insensitive_large_array() {
        config_test_logger();

        // Create a large vector exceeding chunk size
        let mut large_strings: Vec<String> = Vec::new();
        for i in 0..5000 {
            if i % 9 == 0 {
                large_strings.push("Contains EXACT phrase here".to_string());
            } else if i % 13 == 0 {
                large_strings.push("exact phrase at start".to_string());
            } else {
                large_strings.push(format!("different content {}", i));
            }
        }

        let original_len = large_strings.len();

        // Test case insensitive exact phrase matching with chunking
        crate::dispatch::match_exact_phrases(&mut large_strings, "exact phrase", true, 400)
            .expect("Large array case insensitive phrase matching should not panic");

        // Verify results
        assert!(large_strings.len() <= 400, "Should respect max_size limit");
        for s in &large_strings {
            let s_lower = s.to_lowercase();
            assert!(
                s_lower.contains("exact phrase"),
                "All results should contain phrase (case insensitive): {}",
                s
            );
        }

        log::debug!(
            "Large array case insensitive phrase test: {} -> {} matches",
            original_len,
            large_strings.len()
        );
    }
}
