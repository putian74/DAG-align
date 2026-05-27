//! Chunked build and deterministic merge scheduling interfaces.

use crate::algorithms::build::{BuildConfig, FtoDagBuilder};
use crate::algorithms::merge::{MergeConfig, merge_graphs};
use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{ChunkId, GraphId, RoundId, SequenceId};
use crate::graph_model::graph::FtoDag;
use crate::sequence_model::alphabet::Alphabet;
use crate::sequence_model::fragment::FragmentEncoder;
use crate::sequence_model::sequence::{EncodedSequence, SequenceRecord};
use std::thread;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ChunkRange {
    pub chunk_id: ChunkId,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunkPlan {
    pub chunks: Vec<ChunkId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MergeRoundPlan {
    pub round: RoundId,
    pub pairs: Vec<(ChunkId, ChunkId)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MemoryBudget {
    pub max_bytes: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ParallelScratchConfig {
    pub chunk_size: usize,
    pub max_parallel_subgraphs: usize,
    pub memory_budget: Option<MemoryBudget>,
    pub estimated_subgraph_peak_bytes: Option<u64>,
}

impl Default for ParallelScratchConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            max_parallel_subgraphs: 1,
            memory_budget: None,
            estimated_subgraph_peak_bytes: None,
        }
    }
}

impl ParallelScratchConfig {
    fn validate(self) -> Result<Self> {
        if self.chunk_size == 0 {
            return Err(DagError::InvalidRange {
                start: 0,
                end: 0,
                len: 0,
            });
        }
        if self.max_parallel_subgraphs == 0 {
            return Err(DagError::InvalidStorage(
                "max_parallel_subgraphs must be >= 1".to_string(),
            ));
        }
        Ok(self)
    }

    pub fn effective_parallel_subgraphs(self) -> usize {
        let mut effective = self.max_parallel_subgraphs.max(1);
        if let (Some(memory_budget), Some(estimated)) =
            (self.memory_budget, self.estimated_subgraph_peak_bytes)
            && let Some(by_memory) = memory_budget.max_bytes.checked_div(estimated)
        {
            effective = effective.min(by_memory.max(1) as usize);
        }
        effective.max(1)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgressEvent {
    Started(&'static str),
    Advanced { completed: usize, total: usize },
    Finished(&'static str),
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct SubgraphScratchSummary {
    pub chunk_count: usize,
    pub integrated_sequences: usize,
    pub rejected_sequences: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub total_provenance_records: usize,
}

pub trait ProgressSink {
    fn on_progress(&mut self, event: ProgressEvent);
}

#[derive(Clone, Debug, Default)]
pub struct BuildScheduler;

impl BuildScheduler {
    pub fn plan_chunk_ranges(total_sequences: usize, chunk_size: usize) -> Result<Vec<ChunkRange>> {
        if chunk_size == 0 {
            return Err(DagError::InvalidRange {
                start: 0,
                end: 0,
                len: total_sequences,
            });
        }
        let mut ranges = Vec::new();
        let mut start = 0usize;
        while start < total_sequences {
            let end = (start + chunk_size).min(total_sequences);
            ranges.push(ChunkRange {
                chunk_id: ChunkId::try_from(ranges.len())?,
                start,
                end,
            });
            start = end;
        }
        Ok(ranges)
    }

    pub fn plan_chunk_ids(total_sequences: usize, chunk_size: usize) -> Result<ChunkPlan> {
        let ranges = Self::plan_chunk_ranges(total_sequences, chunk_size)?;
        Ok(ChunkPlan {
            chunks: ranges.into_iter().map(|range| range.chunk_id).collect(),
        })
    }

    pub fn plan_merge_rounds(chunk_count: usize) -> Result<Vec<MergeRoundPlan>> {
        if chunk_count == 0 {
            return Ok(Vec::new());
        }
        let mut rounds = Vec::new();
        let mut frontier = (0..chunk_count)
            .map(ChunkId::try_from)
            .collect::<Result<Vec<_>>>()?;
        let mut round_index = 0usize;
        while frontier.len() > 1 {
            let mut pairs = Vec::new();
            let mut next = Vec::with_capacity(frontier.len().div_ceil(2));
            let mut cursor = 0usize;
            while cursor < frontier.len() {
                if cursor + 1 < frontier.len() {
                    pairs.push((frontier[cursor], frontier[cursor + 1]));
                }
                next.push(ChunkId::try_from(next.len())?);
                cursor += 2;
            }
            rounds.push(MergeRoundPlan {
                round: RoundId::try_from(round_index)?,
                pairs,
            });
            frontier = next;
            round_index += 1;
        }
        Ok(rounds)
    }

    fn build_chunk_graph<A, E>(
        chunk_records: &[SequenceRecord],
        alphabet: &A,
        encoder: &E,
        chunk_build_config: BuildConfig,
    ) -> Result<(FtoDag, usize, usize, usize)>
    where
        A: Alphabet + Sync,
        E: FragmentEncoder + Sync,
    {
        let mut builder = FtoDagBuilder::new(chunk_build_config);
        for (index, record) in chunk_records.iter().enumerate() {
            let sequence_id = SequenceId::try_from(index)?;
            let encoded = EncodedSequence::encode(record.clone(), alphabet)?;
            builder.add_sequence_from_encoded_summary(sequence_id, &encoded, encoder)?;
        }
        let report = builder.report().clone();
        let graph = builder.finalize_graph()?;
        Ok((
            graph,
            report.integrated_sequences.len(),
            report.rejected_sequences.len(),
            report.total_provenance_records_added,
        ))
    }

    pub fn build_parallel_subgraph_scratch_only<A, E>(
        records: &[SequenceRecord],
        alphabet: &A,
        encoder: &E,
        build_config: BuildConfig,
        parallel_config: ParallelScratchConfig,
        mut progress: Option<&mut dyn ProgressSink>,
    ) -> Result<SubgraphScratchSummary>
    where
        A: Alphabet + Sync,
        E: FragmentEncoder + Sync,
    {
        let parallel_config = parallel_config.validate()?;
        if records.is_empty() {
            return Ok(SubgraphScratchSummary::default());
        }

        let chunk_ranges = Self::plan_chunk_ranges(records.len(), parallel_config.chunk_size)?;
        let build_batch_width = parallel_config
            .effective_parallel_subgraphs()
            .min(chunk_ranges.len().max(1));
        let total_chunks = chunk_ranges.len();
        if let Some(sink) = &mut progress {
            sink.on_progress(ProgressEvent::Started(
                "parallel_subgraph_scratch_only_build",
            ));
        }

        let mut summary = SubgraphScratchSummary {
            chunk_count: total_chunks,
            ..SubgraphScratchSummary::default()
        };
        let mut completed_chunks = 0usize;

        for batch in chunk_ranges.chunks(build_batch_width) {
            let mut batch_results = Vec::with_capacity(batch.len());
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(batch.len());
                for range in batch.iter().copied() {
                    let chunk_records = &records[range.start..range.end];
                    let mut chunk_build_config = build_config;
                    chunk_build_config.graph_id = GraphId::new(range.chunk_id.raw());
                    handles.push(scope.spawn(
                        move || -> Result<(usize, usize, usize, usize, usize)> {
                            let (graph, integrated, rejected, provenance) =
                                Self::build_chunk_graph(
                                    chunk_records,
                                    alphabet,
                                    encoder,
                                    chunk_build_config,
                                )?;
                            Ok((
                                integrated,
                                rejected,
                                graph.node_count(),
                                graph.edge_count(),
                                provenance,
                            ))
                        },
                    ));
                }
                for handle in handles {
                    batch_results.push(handle.join());
                }
            });

            for join_result in batch_results {
                let (integrated, rejected, nodes, edges, provenance) =
                    join_result.map_err(|_| {
                        DagError::InvalidStorage(
                            "parallel subgraph-only build worker panicked".to_string(),
                        )
                    })??;
                summary.integrated_sequences += integrated;
                summary.rejected_sequences += rejected;
                summary.total_nodes += nodes;
                summary.total_edges += edges;
                summary.total_provenance_records += provenance;
                completed_chunks += 1;
            }
            if let Some(sink) = &mut progress {
                sink.on_progress(ProgressEvent::Advanced {
                    completed: completed_chunks,
                    total: total_chunks,
                });
            }
        }

        if let Some(sink) = &mut progress {
            sink.on_progress(ProgressEvent::Finished(
                "parallel_subgraph_scratch_only_build",
            ));
        }
        Ok(summary)
    }

    pub fn build_parallel_subgraph_scratch<A, E>(
        records: &[SequenceRecord],
        alphabet: &A,
        encoder: &E,
        build_config: BuildConfig,
        merge_config: MergeConfig,
        parallel_config: ParallelScratchConfig,
        mut progress: Option<&mut dyn ProgressSink>,
    ) -> Result<FtoDag>
    where
        A: Alphabet + Sync,
        E: FragmentEncoder + Sync,
    {
        let parallel_config = parallel_config.validate()?;
        if records.is_empty() {
            return FtoDagBuilder::new(build_config).finalize_graph();
        }

        let chunk_ranges = Self::plan_chunk_ranges(records.len(), parallel_config.chunk_size)?;
        let build_batch_width = parallel_config
            .effective_parallel_subgraphs()
            .min(chunk_ranges.len().max(1));
        let total_chunks = chunk_ranges.len();
        if let Some(sink) = &mut progress {
            sink.on_progress(ProgressEvent::Started("parallel_subgraph_scratch_build"));
        }

        let mut built_chunks = vec![None; chunk_ranges.len()];
        let mut completed_chunks = 0usize;
        for batch in chunk_ranges.chunks(build_batch_width) {
            let mut batch_results = Vec::with_capacity(batch.len());
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(batch.len());
                for range in batch.iter().copied() {
                    let chunk_records = &records[range.start..range.end];
                    let mut chunk_build_config = build_config;
                    chunk_build_config.graph_id = GraphId::new(range.chunk_id.raw());
                    handles.push((
                        range.chunk_id.to_usize(),
                        scope.spawn(move || -> Result<FtoDag> {
                            Self::build_chunk_graph(
                                chunk_records,
                                alphabet,
                                encoder,
                                chunk_build_config,
                            )
                            .map(|(graph, _, _, _)| graph)
                        }),
                    ));
                }
                for (chunk_index, handle) in handles {
                    batch_results.push((chunk_index, handle.join()));
                }
            });
            for (chunk_index, join_result) in batch_results {
                let chunk_graph = join_result.map_err(|_| {
                    DagError::InvalidStorage("parallel subgraph build worker panicked".to_string())
                })??;
                built_chunks[chunk_index] = Some(chunk_graph);
                completed_chunks += 1;
            }
            if let Some(sink) = &mut progress {
                sink.on_progress(ProgressEvent::Advanced {
                    completed: completed_chunks,
                    total: total_chunks,
                });
            }
        }

        let mut frontier = built_chunks
            .into_iter()
            .map(|graph| graph.expect("all chunk graphs are built"))
            .collect::<Vec<_>>();
        let mut merge_round = 0usize;
        let total_merge_rounds = Self::plan_merge_rounds(frontier.len())?.len();
        while frontier.len() > 1 {
            let mut next = Vec::with_capacity(frontier.len().div_ceil(2));
            let mut iter = frontier.into_iter();
            while let Some(left) = iter.next() {
                if let Some(right) = iter.next() {
                    next.push(merge_graphs(left, right, merge_config)?);
                } else {
                    next.push(left);
                }
            }
            frontier = next;
            merge_round += 1;
            if let Some(sink) = &mut progress {
                sink.on_progress(ProgressEvent::Advanced {
                    completed: merge_round,
                    total: total_merge_rounds,
                });
            }
        }
        if let Some(sink) = &mut progress {
            sink.on_progress(ProgressEvent::Finished("parallel_subgraph_scratch_build"));
        }
        frontier.pop().ok_or_else(|| {
            DagError::InvalidStorage("parallel scratch build produced no graph".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::build::BuildConfig;
    use crate::graph_model::provenance::ProvenanceStorageStrategy;
    use crate::graph_model::validate::ValidateGraph;
    use crate::sequence_model::alphabet::BuiltinAlphabet;
    use crate::sequence_model::fragment::DefaultFragmentEncoder;
    use crate::sequence_model::sequence::VecSequenceInput;

    #[test]
    fn plan_chunk_ranges_splits_records_into_fixed_windows() {
        let ranges = BuildScheduler::plan_chunk_ranges(10, 3).unwrap();
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, 3);
        assert_eq!(ranges[1].start, 3);
        assert_eq!(ranges[1].end, 6);
        assert_eq!(ranges[2].start, 6);
        assert_eq!(ranges[2].end, 9);
        assert_eq!(ranges[3].start, 9);
        assert_eq!(ranges[3].end, 10);
    }

    #[test]
    fn parallel_config_applies_memory_limited_parallelism() {
        let config = ParallelScratchConfig {
            chunk_size: 4,
            max_parallel_subgraphs: 8,
            memory_budget: Some(MemoryBudget { max_bytes: 300 }),
            estimated_subgraph_peak_bytes: Some(128),
        };
        assert_eq!(config.effective_parallel_subgraphs(), 2);
    }

    #[test]
    fn parallel_subgraph_scratch_only_reports_chunk_aggregate_stats() {
        let records = vec![
            SequenceRecord::new("s0", "ACGTACGT"),
            SequenceRecord::new("s1", "ACGTACGA"),
            SequenceRecord::new("s2", "ACGTTCGT"),
            SequenceRecord::new("s3", "ACGTACGG"),
            SequenceRecord::new("s4", "ACGTACGC"),
            SequenceRecord::new("s5", "ACGTACGT"),
        ];
        let alphabet = BuiltinAlphabet::dna_canonical();
        let encoder = DefaultFragmentEncoder::general();
        let mut build_config = BuildConfig::new(3);
        build_config.provenance_storage_strategy = ProvenanceStorageStrategy::TracePaths;

        let summary = BuildScheduler::build_parallel_subgraph_scratch_only(
            &records,
            &alphabet,
            &encoder,
            build_config,
            ParallelScratchConfig {
                chunk_size: 2,
                max_parallel_subgraphs: 2,
                memory_budget: None,
                estimated_subgraph_peak_bytes: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(summary.chunk_count, 3);
        assert_eq!(
            summary.integrated_sequences + summary.rejected_sequences,
            records.len()
        );
        assert!(summary.total_nodes > 0);
        assert!(summary.total_edges > 0);
        assert!(summary.total_provenance_records > 0);
    }

    #[test]
    fn parallel_subgraph_scratch_matches_serial_baseline() {
        let records = vec![
            SequenceRecord::new("s0", "ACGTACGT"),
            SequenceRecord::new("s1", "ACGTACGA"),
            SequenceRecord::new("s2", "ACGTTCGT"),
            SequenceRecord::new("s3", "ACGTACGG"),
            SequenceRecord::new("s4", "ACGTACGC"),
            SequenceRecord::new("s5", "ACGTACGT"),
        ];
        let alphabet = BuiltinAlphabet::dna_canonical();
        let encoder = DefaultFragmentEncoder::general();
        let mut build_config = BuildConfig::new(3);
        build_config.provenance_storage_strategy = ProvenanceStorageStrategy::TracePaths;

        let mut serial_builder = FtoDagBuilder::new(build_config);
        let mut serial_input = VecSequenceInput::new(records.clone());
        serial_builder
            .build_from_input(&mut serial_input, &alphabet, &encoder)
            .unwrap();
        let serial = serial_builder.finalize_graph().unwrap();

        let parallel = BuildScheduler::build_parallel_subgraph_scratch(
            &records,
            &alphabet,
            &encoder,
            build_config,
            MergeConfig::default(),
            ParallelScratchConfig {
                chunk_size: 2,
                max_parallel_subgraphs: 2,
                memory_budget: None,
                estimated_subgraph_peak_bytes: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(parallel.nodes(), serial.nodes());
        assert_eq!(parallel.edges(), serial.edges());
        assert_eq!(parallel.node_count(), serial.node_count());
        assert_eq!(parallel.edge_count(), serial.edge_count());
        assert!(parallel.validate().is_valid());
    }
}
