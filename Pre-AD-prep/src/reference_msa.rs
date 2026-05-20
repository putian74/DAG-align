//! Reference-path MSA preprocessing and PHMM state-band derivation.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use crate::coordinates::ReferencePath;
use crate::error::{PreAdPrepError, Result};
use crate::graph::TensorGraph;

const STATE_MATCH: u8 = 0;
const STATE_DELETE: u8 = 1;
const STATE_INSERT: u8 = 2;
const NEG_INF: i32 = i32::MIN / 4;
const BASIS_POINTS_PER_ONE: u64 = 10_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceMsaScoring {
    pub alphabet_size: usize,
    pub substitution: Vec<i32>,
    pub gap_open: i32,
    pub gap_extend: i32,
}

impl ReferenceMsaScoring {
    pub fn identity(
        alphabet_size: usize,
        match_score: i32,
        mismatch_score: i32,
        gap_open: i32,
        gap_extend: i32,
    ) -> Self {
        let mut substitution = vec![mismatch_score; alphabet_size.saturating_mul(alphabet_size)];
        for symbol in 0..alphabet_size {
            substitution[symbol * alphabet_size + symbol] = match_score;
        }
        Self {
            alphabet_size,
            substitution,
            gap_open,
            gap_extend,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.alphabet_size == 0 {
            return Err(PreAdPrepError::Validation(
                "reference MSA scoring alphabet size must be positive".into(),
            ));
        }
        if self.substitution.len() != self.alphabet_size.saturating_mul(self.alphabet_size) {
            return Err(PreAdPrepError::Validation(format!(
                "reference MSA substitution matrix length {} does not match alphabet size {}",
                self.substitution.len(),
                self.alphabet_size
            )));
        }
        Ok(())
    }

    fn score(&self, left: u16, right: u16) -> Result<i32> {
        let left = left as usize;
        let right = right as usize;
        if left >= self.alphabet_size || right >= self.alphabet_size {
            return Err(PreAdPrepError::Validation(format!(
                "reference MSA symbol {} or {} exceeds alphabet size {}",
                left, right, self.alphabet_size
            )));
        }
        Ok(self.substitution[left * self.alphabet_size + right])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceMsaConfig {
    pub scoring: ReferenceMsaScoring,
    pub minimum_gap_threshold_basis_points: u16,
    pub canonical_gap_threshold_basis_points: u16,
    pub maximum_gap_threshold_basis_points: u16,
}

impl Default for ReferenceMsaConfig {
    fn default() -> Self {
        Self {
            scoring: ReferenceMsaScoring::identity(16, 2, -1, -5, -1),
            minimum_gap_threshold_basis_points: 3_000,
            canonical_gap_threshold_basis_points: 5_000,
            maximum_gap_threshold_basis_points: 7_000,
        }
    }
}

impl ReferenceMsaConfig {
    pub fn validate(&self) -> Result<()> {
        self.scoring.validate()?;
        if self.minimum_gap_threshold_basis_points > self.canonical_gap_threshold_basis_points
            || self.canonical_gap_threshold_basis_points > self.maximum_gap_threshold_basis_points
            || self.maximum_gap_threshold_basis_points > BASIS_POINTS_PER_ONE as u16
        {
            return Err(PreAdPrepError::Validation(
                "reference MSA gap thresholds must satisfy minimum <= canonical <= maximum <= 10000"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightedSequencePath {
    pub representative_sequence_id: usize,
    pub sequence_ids: Vec<usize>,
    pub node_ids: Vec<usize>,
    pub symbols: Vec<u16>,
    pub weight: u64,
}

impl WeightedSequencePath {
    pub fn validate(&self) -> Result<()> {
        if self.sequence_ids.is_empty() {
            return Err(PreAdPrepError::Validation(
                "reference MSA path must include at least one sequence id".into(),
            ));
        }
        if self.node_ids.len() != self.symbols.len() {
            return Err(PreAdPrepError::Validation(
                "reference MSA path node_ids and symbols must have equal lengths".into(),
            ));
        }
        if self.weight == 0 {
            return Err(PreAdPrepError::Validation(
                "reference MSA path weight must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PathAlignmentOp {
    Match {
        reference_index: usize,
        query_offset: usize,
        query_node_id: usize,
    },
    Delete {
        reference_index: usize,
    },
    Insert {
        anchor_index: usize,
        slot: usize,
        query_offset: usize,
        query_node_id: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathToReferenceAlignment {
    pub sequence_ids: Vec<usize>,
    pub weight: u64,
    pub score: i32,
    pub operations: Vec<PathAlignmentOp>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceMsaColumnKind {
    Insert {
        anchor_index: usize,
        slot: usize,
    },
    Match {
        reference_index: usize,
        reference_node_id: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceMsaColumn {
    pub kind: ReferenceMsaColumnKind,
    pub gap_weight: u64,
    pub gap_fraction_basis_points: u16,
    pub symbol_weights: Vec<(u16, u64)>,
    pub node_weights: Vec<(usize, u64)>,
    pub primary_node_id: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateSelection {
    pub included_column_indices: Vec<usize>,
    pub column_to_state: Vec<Option<usize>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceStateBands {
    pub minimum: StateSelection,
    pub canonical: StateSelection,
    pub maximum: StateSelection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceMsaResult {
    pub reference_path: ReferencePath,
    pub reference_symbols: Vec<u16>,
    pub total_weight: u64,
    pub longest_insertions: Vec<usize>,
    pub columns: Vec<ReferenceMsaColumn>,
    pub reference_column_indices: Vec<usize>,
    pub insertion_anchor_column_ranges: Vec<(usize, usize)>,
    pub path_alignments: Vec<PathToReferenceAlignment>,
    pub state_bands: ReferenceStateBands,
}

#[derive(Clone, Default)]
struct ColumnAccumulator {
    symbol_weights: BTreeMap<u16, u64>,
    node_weights: BTreeMap<usize, u64>,
}

pub fn select_max_weight_reference_path(paths: &[WeightedSequencePath]) -> Result<usize> {
    let Some((index, _)) = paths
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_reference_candidates(left, right))
    else {
        return Err(PreAdPrepError::Validation(
            "cannot select a reference path from an empty path list".into(),
        ));
    };
    Ok(index)
}

pub fn collect_weighted_sequence_paths(
    graph: &TensorGraph,
    source_sequence_id: &[u64],
    source_position: &[u64],
    node_source_offset: &[usize],
    node_source_len: &[usize],
) -> Result<Vec<WeightedSequencePath>> {
    if source_sequence_id.len() != source_position.len() {
        return Err(PreAdPrepError::Validation(
            "source_sequence_id and source_position lengths must match".into(),
        ));
    }
    if node_source_offset.len() != graph.node_count() || node_source_len.len() != graph.node_count()
    {
        return Err(PreAdPrepError::Validation(
            "node source range arrays must match graph.node_count".into(),
        ));
    }

    let mut topo_rank = vec![0usize; graph.node_count()];
    for (rank, &node_id) in graph.topo_order.iter().enumerate() {
        topo_rank[node_id] = rank;
    }

    let mut per_sequence: HashMap<usize, Vec<(u64, usize, usize)>> = HashMap::new();
    for node_id in 0..graph.node_count() {
        let start = node_source_offset[node_id];
        let len = node_source_len[node_id];
        let end = start.saturating_add(len);
        if end > source_sequence_id.len() {
            return Err(PreAdPrepError::Validation(format!(
                "node source range [{start}, {end}) exceeds decoded source record count {}",
                source_sequence_id.len()
            )));
        }
        for record_index in start..end {
            let sequence_id = usize::try_from(source_sequence_id[record_index]).map_err(|_| {
                PreAdPrepError::Validation(format!(
                    "source sequence id {} cannot fit into usize",
                    source_sequence_id[record_index]
                ))
            })?;
            per_sequence.entry(sequence_id).or_default().push((
                source_position[record_index],
                topo_rank[node_id],
                node_id,
            ));
        }
    }

    let mut grouped: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (sequence_id, entries) in per_sequence {
        let mut entries = entries;
        entries.sort_by(|left, right| left.cmp(right));
        let mut path = Vec::with_capacity(entries.len());
        for (_, _, node_id) in entries {
            if path.last().copied() != Some(node_id) {
                path.push(node_id);
            }
        }
        if !path.is_empty() {
            grouped.entry(path).or_default().push(sequence_id);
        }
    }

    let mut paths = Vec::with_capacity(grouped.len());
    for (node_ids, sequence_ids) in grouped {
        let symbols = node_ids
            .iter()
            .map(|&node_id| graph.node_symbol[node_id])
            .collect();
        let path = WeightedSequencePath {
            representative_sequence_id: sequence_ids[0],
            sequence_ids: sequence_ids.clone(),
            node_ids,
            symbols,
            weight: sequence_ids.len() as u64,
        };
        path.validate()?;
        paths.push(path);
    }
    Ok(paths)
}

pub fn build_reference_msa_against_reference(
    paths: &[WeightedSequencePath],
    reference_path: &ReferencePath,
    reference_symbols: &[u16],
    config: &ReferenceMsaConfig,
) -> Result<ReferenceMsaResult> {
    let reference_node_ids = reference_path
        .node_ids
        .iter()
        .map(|node_id| {
            node_id.ok_or_else(|| {
                PreAdPrepError::Validation(
                    "reference MSA initialization requires a fully materialized reference path"
                        .into(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    build_reference_msa_from_reference(
        paths,
        reference_path.clone(),
        reference_node_ids,
        reference_symbols.to_vec(),
        config,
    )
}

pub fn build_reference_msa(
    paths: &[WeightedSequencePath],
    reference_index: usize,
    config: &ReferenceMsaConfig,
) -> Result<ReferenceMsaResult> {
    config.validate()?;
    if reference_index >= paths.len() {
        return Err(PreAdPrepError::Validation(format!(
            "reference path index {} exceeds path count {}",
            reference_index,
            paths.len()
        )));
    }
    for path in paths {
        path.validate()?;
    }

    let reference = &paths[reference_index];
    build_reference_msa_from_reference(
        paths,
        ReferencePath {
            node_ids: reference.node_ids.iter().copied().map(Some).collect(),
        },
        reference.node_ids.clone(),
        reference.symbols.clone(),
        config,
    )
}

pub fn insertion_ranges(longest_insertions: &[usize]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (anchor, &length) in longest_insertions.iter().enumerate() {
        if length > 0 {
            if start.is_none() {
                start = Some(anchor);
            }
        } else if let Some(left) = start.take() {
            ranges.push((left, anchor));
        }
    }
    if let Some(left) = start {
        ranges.push((left, longest_insertions.len()));
    }
    ranges
}

pub fn emission_probability_tables(
    result: &ReferenceMsaResult,
    alphabet_size: usize,
    pseudocount: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if alphabet_size == 0 {
        return Err(PreAdPrepError::Validation(
            "reference MSA emission alphabet size must be positive".into(),
        ));
    }
    if !pseudocount.is_finite() || pseudocount <= 0.0 {
        return Err(PreAdPrepError::Validation(
            "reference MSA pseudocount must be positive and finite".into(),
        ));
    }

    let state_count = result.reference_path.global_state_count();
    let mut match_emission = vec![0.0_f64; state_count * alphabet_size];
    for (reference_index, &column_index) in result.reference_column_indices.iter().enumerate() {
        let column = &result.columns[column_index];
        let mut counts = vec![pseudocount; alphabet_size];
        for &(symbol, weight) in &column.symbol_weights {
            let symbol = symbol as usize;
            if symbol >= alphabet_size {
                return Err(PreAdPrepError::Validation(format!(
                    "match-column symbol {} exceeds alphabet size {}",
                    symbol, alphabet_size
                )));
            }
            counts[symbol] += weight as f64;
        }
        let total = counts.iter().sum::<f64>();
        for symbol in 0..alphabet_size {
            match_emission[reference_index * alphabet_size + symbol] = counts[symbol] / total;
        }
    }

    let mut background = vec![pseudocount; alphabet_size];
    for column in &result.columns {
        for &(symbol, weight) in &column.symbol_weights {
            let symbol = symbol as usize;
            if symbol < alphabet_size {
                background[symbol] += weight as f64;
            }
        }
    }
    let background_total = background.iter().sum::<f64>();
    for value in &mut background {
        *value /= background_total;
    }

    let mut insert_emission = vec![0.0_f64; (state_count + 1) * alphabet_size];
    for anchor_index in 0..=state_count {
        let (start, end) = result.insertion_anchor_column_ranges[anchor_index];
        let mut counts = vec![pseudocount; alphabet_size];
        for column in &result.columns[start..end] {
            for &(symbol, weight) in &column.symbol_weights {
                let symbol = symbol as usize;
                if symbol >= alphabet_size {
                    return Err(PreAdPrepError::Validation(format!(
                        "insert-column symbol {} exceeds alphabet size {}",
                        symbol, alphabet_size
                    )));
                }
                counts[symbol] += weight as f64;
            }
        }
        let total = counts.iter().sum::<f64>();
        let row_offset = anchor_index * alphabet_size;
        if start == end {
            insert_emission[row_offset..row_offset + alphabet_size].copy_from_slice(&background);
        } else {
            for symbol in 0..alphabet_size {
                insert_emission[row_offset + symbol] = counts[symbol] / total;
            }
        }
    }

    Ok((match_emission, insert_emission))
}

fn build_reference_msa_from_reference(
    paths: &[WeightedSequencePath],
    reference_path: ReferencePath,
    reference_node_ids: Vec<usize>,
    reference_symbols: Vec<u16>,
    config: &ReferenceMsaConfig,
) -> Result<ReferenceMsaResult> {
    config.validate()?;
    for path in paths {
        path.validate()?;
    }

    let ref_len = reference_symbols.len();
    if reference_node_ids.len() != ref_len {
        return Err(PreAdPrepError::Validation(
            "reference node id count and reference symbol count must match".into(),
        ));
    }
    let reference = WeightedSequencePath {
        representative_sequence_id: usize::MAX,
        sequence_ids: vec![usize::MAX],
        node_ids: reference_node_ids,
        symbols: reference_symbols.clone(),
        weight: 1,
    };
    let total_weight = paths.iter().map(|path| path.weight).sum::<u64>();
    let mut longest_insertions = vec![0usize; ref_len + 1];
    let mut match_accumulators = vec![ColumnAccumulator::default(); ref_len];
    let mut insert_accumulators = vec![Vec::<ColumnAccumulator>::new(); ref_len + 1];
    let mut path_alignments = Vec::with_capacity(paths.len());

    for path in paths {
        let alignment = align_path_to_reference(&reference, path, &config.scoring)?;
        for operation in &alignment.operations {
            match *operation {
                PathAlignmentOp::Match {
                    reference_index,
                    query_offset,
                    query_node_id,
                } => {
                    *match_accumulators[reference_index]
                        .symbol_weights
                        .entry(path.symbols[query_offset])
                        .or_default() += path.weight;
                    *match_accumulators[reference_index]
                        .node_weights
                        .entry(query_node_id)
                        .or_default() += path.weight;
                }
                PathAlignmentOp::Delete { .. } => {}
                PathAlignmentOp::Insert {
                    anchor_index,
                    slot,
                    query_offset,
                    query_node_id,
                } => {
                    while insert_accumulators[anchor_index].len() <= slot {
                        insert_accumulators[anchor_index].push(ColumnAccumulator::default());
                    }
                    *insert_accumulators[anchor_index][slot]
                        .symbol_weights
                        .entry(path.symbols[query_offset])
                        .or_default() += path.weight;
                    *insert_accumulators[anchor_index][slot]
                        .node_weights
                        .entry(query_node_id)
                        .or_default() += path.weight;
                    longest_insertions[anchor_index] =
                        longest_insertions[anchor_index].max(slot + 1);
                }
            }
        }
        path_alignments.push(alignment);
    }

    let mut columns = Vec::new();
    let mut reference_column_indices = Vec::with_capacity(ref_len);
    let mut insertion_anchor_column_ranges = Vec::with_capacity(ref_len + 1);
    for anchor_index in 0..=ref_len {
        let start = columns.len();
        for (slot, accumulator) in insert_accumulators[anchor_index].iter().enumerate() {
            columns.push(finalize_column(
                ReferenceMsaColumnKind::Insert { anchor_index, slot },
                accumulator,
                total_weight,
            ));
        }
        insertion_anchor_column_ranges.push((start, columns.len()));
        if anchor_index < ref_len {
            reference_column_indices.push(columns.len());
            columns.push(finalize_column(
                ReferenceMsaColumnKind::Match {
                    reference_index: anchor_index,
                    reference_node_id: reference.node_ids[anchor_index],
                },
                &match_accumulators[anchor_index],
                total_weight,
            ));
        }
    }

    Ok(ReferenceMsaResult {
        reference_path,
        reference_symbols,
        total_weight,
        longest_insertions,
        columns: columns.clone(),
        reference_column_indices,
        insertion_anchor_column_ranges,
        path_alignments,
        state_bands: ReferenceStateBands {
            minimum: select_columns_by_gap_threshold(
                &columns,
                config.minimum_gap_threshold_basis_points,
            ),
            canonical: select_columns_by_gap_threshold(
                &columns,
                config.canonical_gap_threshold_basis_points,
            ),
            maximum: select_columns_by_gap_threshold(
                &columns,
                config.maximum_gap_threshold_basis_points,
            ),
        },
    })
}

fn compare_reference_candidates(
    left: &WeightedSequencePath,
    right: &WeightedSequencePath,
) -> Ordering {
    left.weight
        .cmp(&right.weight)
        .then_with(|| left.node_ids.len().cmp(&right.node_ids.len()))
        .then_with(|| {
            right
                .representative_sequence_id
                .cmp(&left.representative_sequence_id)
        })
}

fn finalize_column(
    kind: ReferenceMsaColumnKind,
    accumulator: &ColumnAccumulator,
    total_weight: u64,
) -> ReferenceMsaColumn {
    let nongap_weight = accumulator.symbol_weights.values().copied().sum::<u64>();
    let gap_weight = total_weight.saturating_sub(nongap_weight);
    let primary_node_id = accumulator
        .node_weights
        .iter()
        .max_by(|(left_node, left_weight), (right_node, right_weight)| {
            left_weight
                .cmp(right_weight)
                .then_with(|| right_node.cmp(left_node))
        })
        .map(|(node_id, _)| *node_id);
    ReferenceMsaColumn {
        kind,
        gap_weight,
        gap_fraction_basis_points: gap_fraction_basis_points(gap_weight, total_weight),
        symbol_weights: accumulator
            .symbol_weights
            .iter()
            .map(|(symbol, weight)| (*symbol, *weight))
            .collect(),
        node_weights: accumulator
            .node_weights
            .iter()
            .map(|(node_id, weight)| (*node_id, *weight))
            .collect(),
        primary_node_id,
    }
}

fn select_columns_by_gap_threshold(
    columns: &[ReferenceMsaColumn],
    gap_threshold_basis_points: u16,
) -> StateSelection {
    let mut included_column_indices = Vec::new();
    let mut column_to_state = vec![None; columns.len()];
    for (column_index, column) in columns.iter().enumerate() {
        if column.gap_fraction_basis_points < gap_threshold_basis_points {
            let state_id = included_column_indices.len();
            included_column_indices.push(column_index);
            column_to_state[column_index] = Some(state_id);
        }
    }
    StateSelection {
        included_column_indices,
        column_to_state,
    }
}

fn gap_fraction_basis_points(gap_weight: u64, total_weight: u64) -> u16 {
    if total_weight == 0 {
        return 0;
    }
    let scaled = gap_weight
        .saturating_mul(BASIS_POINTS_PER_ONE)
        .saturating_add(total_weight / 2)
        / total_weight;
    scaled.min(BASIS_POINTS_PER_ONE) as u16
}

fn align_path_to_reference(
    reference: &WeightedSequencePath,
    query: &WeightedSequencePath,
    scoring: &ReferenceMsaScoring,
) -> Result<PathToReferenceAlignment> {
    let ref_len = reference.symbols.len();
    let query_len = query.symbols.len();
    let stride = query_len + 1;
    let cell_count = (ref_len + 1).saturating_mul(stride);
    let mut match_dp = vec![NEG_INF; cell_count];
    let mut delete_dp = vec![NEG_INF; cell_count];
    let mut insert_dp = vec![NEG_INF; cell_count];
    let mut match_bt = vec![STATE_MATCH; cell_count];
    let mut delete_bt = vec![STATE_MATCH; cell_count];
    let mut insert_bt = vec![STATE_MATCH; cell_count];

    match_dp[0] = 0;
    for i in 1..=ref_len {
        let index = i * stride;
        delete_dp[index] = scoring
            .gap_open
            .saturating_add(scoring.gap_extend.saturating_mul(i as i32));
        delete_bt[index] = STATE_DELETE;
    }
    for j in 1..=query_len {
        insert_dp[j] = scoring
            .gap_open
            .saturating_add(scoring.gap_extend.saturating_mul(j as i32));
        insert_bt[j] = STATE_INSERT;
    }

    for i in 1..=ref_len {
        for j in 1..=query_len {
            let index = i * stride + j;
            let diag_index = (i - 1) * stride + (j - 1);
            let up_index = (i - 1) * stride + j;
            let left_index = i * stride + (j - 1);

            let score = scoring.score(reference.symbols[i - 1], query.symbols[j - 1])?;
            let (match_state, match_prev) = max3(
                match_dp[diag_index],
                delete_dp[diag_index],
                insert_dp[diag_index],
            );
            match_dp[index] = match_prev.saturating_add(score);
            match_bt[index] = match_state;

            let delete_from_match = match_dp[up_index]
                .saturating_add(scoring.gap_open)
                .saturating_add(scoring.gap_extend);
            let delete_from_delete = delete_dp[up_index].saturating_add(scoring.gap_extend);
            let delete_from_insert = insert_dp[up_index]
                .saturating_add(scoring.gap_open)
                .saturating_add(scoring.gap_extend);
            let (delete_state, delete_prev) =
                max3(delete_from_match, delete_from_delete, delete_from_insert);
            delete_dp[index] = delete_prev;
            delete_bt[index] = delete_state;

            let insert_from_match = match_dp[left_index]
                .saturating_add(scoring.gap_open)
                .saturating_add(scoring.gap_extend);
            let insert_from_delete = delete_dp[left_index]
                .saturating_add(scoring.gap_open)
                .saturating_add(scoring.gap_extend);
            let insert_from_insert = insert_dp[left_index].saturating_add(scoring.gap_extend);
            let (insert_state, insert_prev) =
                max3(insert_from_match, insert_from_delete, insert_from_insert);
            insert_dp[index] = insert_prev;
            insert_bt[index] = insert_state;
        }
    }

    let final_index = ref_len * stride + query_len;
    let (mut state, score) = max3(
        match_dp[final_index],
        delete_dp[final_index],
        insert_dp[final_index],
    );
    let mut i = ref_len;
    let mut j = query_len;
    let mut operations = Vec::with_capacity(ref_len + query_len);
    while i > 0 || j > 0 {
        let index = i * stride + j;
        match state {
            STATE_MATCH => {
                if i == 0 || j == 0 {
                    return Err(PreAdPrepError::Validation(
                        "reference MSA traceback entered match state on a border cell".into(),
                    ));
                }
                let previous_state = match_bt[index];
                operations.push(PathAlignmentOp::Match {
                    reference_index: i - 1,
                    query_offset: j - 1,
                    query_node_id: query.node_ids[j - 1],
                });
                i -= 1;
                j -= 1;
                state = previous_state;
            }
            STATE_DELETE => {
                if i == 0 {
                    return Err(PreAdPrepError::Validation(
                        "reference MSA traceback entered delete state on the top border".into(),
                    ));
                }
                let previous_state = delete_bt[index];
                operations.push(PathAlignmentOp::Delete {
                    reference_index: i - 1,
                });
                i -= 1;
                state = previous_state;
            }
            STATE_INSERT => {
                if j == 0 {
                    return Err(PreAdPrepError::Validation(
                        "reference MSA traceback entered insert state on the left border".into(),
                    ));
                }
                let previous_state = insert_bt[index];
                operations.push(PathAlignmentOp::Insert {
                    anchor_index: i,
                    slot: 0,
                    query_offset: j - 1,
                    query_node_id: query.node_ids[j - 1],
                });
                j -= 1;
                state = previous_state;
            }
            _ => {
                return Err(PreAdPrepError::Validation(
                    "reference MSA traceback encountered an unknown DP state".into(),
                ));
            }
        }
    }
    operations.reverse();
    assign_insertion_slots(&mut operations);

    Ok(PathToReferenceAlignment {
        sequence_ids: query.sequence_ids.clone(),
        weight: query.weight,
        score,
        operations,
    })
}

fn assign_insertion_slots(operations: &mut [PathAlignmentOp]) {
    let mut current_anchor = None;
    let mut next_slot = 0usize;
    for operation in operations {
        match operation {
            PathAlignmentOp::Insert {
                anchor_index, slot, ..
            } => {
                if current_anchor == Some(*anchor_index) {
                    *slot = next_slot;
                    next_slot += 1;
                } else {
                    current_anchor = Some(*anchor_index);
                    next_slot = 1;
                    *slot = 0;
                }
            }
            PathAlignmentOp::Match { .. } | PathAlignmentOp::Delete { .. } => {
                current_anchor = None;
                next_slot = 0;
            }
        }
    }
}

fn max3(a: i32, b: i32, c: i32) -> (u8, i32) {
    if a >= b && a >= c {
        (STATE_MATCH, a)
    } else if b >= c {
        (STATE_DELETE, b)
    } else {
        (STATE_INSERT, c)
    }
}
