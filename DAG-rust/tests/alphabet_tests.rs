use dag_rust::prelude::*;

#[test]
fn dna_exact_and_canonical_policies_are_distinct() {
    let exact = BuiltinAlphabet::dna_exact();
    assert_eq!(exact.bits_per_symbol(), 4);
    assert!(exact.encode('N').is_ok());
    assert_eq!(
        exact.decode(exact.encode('n').expect("case-folded N")),
        Some('N')
    );

    let canonical = BuiltinAlphabet::dna_canonical();
    assert_eq!(canonical.bits_per_symbol(), 2);
    assert!(canonical.encode('A').is_ok());
    assert!(canonical.encode('N').is_err());
}

#[test]
fn rna_t_u_policy_is_explicit() {
    let strict_rna = BuiltinAlphabet::rna_exact(NormalizationPolicy::CaseFold);
    assert!(strict_rna.encode('U').is_ok());
    assert!(strict_rna.encode('T').is_err());

    let dna_to_rna = BuiltinAlphabet::rna_exact(NormalizationPolicy::DnaToRna);
    assert_eq!(
        dna_to_rna.decode(dna_to_rna.encode('T').unwrap()),
        Some('U')
    );
}

#[test]
fn protein_extended_symbols_fit_in_five_bits() {
    let protein = BuiltinAlphabet::protein_exact();
    assert_eq!(protein.bits_per_symbol(), 5);
    for symbol in ['A', 'Y', 'B', 'J', 'Z', 'X', '*'] {
        assert!(protein.encode(symbol).is_ok(), "{symbol} should encode");
    }
}

#[test]
fn custom_alphabet_preserves_explicit_symbol_set() {
    let custom = CustomAlphabet::new(['@', '#', '$']).expect("custom symbols");
    let hash = custom.encode('#').expect("hash encodes");
    assert_eq!(custom.decode(hash), Some('#'));
    assert_eq!(custom.symbol_count(), 3);
    assert_eq!(custom.bits_per_symbol(), 2);
    assert!(custom.encode('%').is_err());
}
