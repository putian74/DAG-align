use dag_rust::prelude::*;

fn encoded_repeated_a(len: usize) -> EncodedSequence {
    EncodedSequence::from_symbols("repeat-a", vec![SymbolId::new(0); len])
}

#[test]
fn packed_inline_uses_u128_capacity_boundary() {
    let sequence = encoded_repeated_a(33);
    let encoder = DefaultFragmentEncoder::packed(BitWidth::new(4).unwrap());

    let key_32 = encoder
        .encode_window(&sequence.symbols()[..32])
        .expect("32 4-bit symbols fit");
    assert!(matches!(key_32, FragmentKey::PackedInline { len: 32, .. }));
    assert_eq!(key_32.len(), 32);
    assert_eq!(key_32.bits_per_symbol(), Some(4));

    let key_33 = encoder
        .encode_window(&sequence.symbols()[..33])
        .expect("oversized inline windows fall back");
    assert!(matches!(key_33, FragmentKey::Symbols(_)));
    assert_eq!(key_33.len(), 33);
}

#[test]
fn fragment_occurrences_mark_path_positions() {
    let alphabet = BuiltinAlphabet::dna_exact();
    let sequence = EncodedSequence::encode(SequenceRecord::new("seq", "ACGT"), &alphabet).unwrap();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());

    let occurrences = FragmentOccurrences::new(&sequence, 2, &encoder)
        .expect("valid windows")
        .collect::<Result<Vec<_>>>()
        .expect("occurrences encode");

    assert_eq!(occurrences.len(), 3);
    assert_eq!(occurrences[0].kind, PathPositionKind::Start);
    assert_eq!(occurrences[1].kind, PathPositionKind::Internal);
    assert_eq!(occurrences[2].kind, PathPositionKind::End);
    assert_eq!(occurrences[2].position, 2);
}

#[test]
fn single_window_sequence_is_singleton() {
    let sequence = encoded_repeated_a(2);
    let encoder = DefaultFragmentEncoder::packed(BitWidth::new(2).unwrap());
    let occurrences = FragmentOccurrences::new(&sequence, 2, &encoder)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();

    assert_eq!(occurrences.len(), 1);
    assert_eq!(occurrences[0].kind, PathPositionKind::Singleton);
}

#[test]
fn rolling_packed_occurrences_match_window_encoder() {
    let alphabet = BuiltinAlphabet::dna_exact();
    let sequence =
        EncodedSequence::encode(SequenceRecord::new("seq", "ACGTNNACGTAC"), &alphabet).unwrap();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());

    let window_occurrences = FragmentOccurrences::new(&sequence, 5, &encoder)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let rolling_occurrences = encoder.encode_occurrences(&sequence, 5).unwrap();

    assert_eq!(rolling_occurrences, window_occurrences);
}
