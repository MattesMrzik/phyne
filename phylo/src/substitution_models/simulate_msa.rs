//! Simulation of alignments and ancestral alignments by substitutions only.
//!
//! This module provides the [`SubstitutionSimulator`] which evolves characters down a
//! [`Tree`](`crate::tree::Tree`) under a substitution model, producing a
//! [MASA](`crate::alignment::MASA`) without any gaps.
//!
//! # Example
//! ```
//! use phylo::alignment::{Alignment, AlignmentSimulation, MASA};
//! use phylo::random::DefaultGenerator;
//! use phylo::substitution_models::{dna_models::GTR, SubstModel, SubstitutionSimulator};
//! use phylo::tree;
//! # use phylo::Result;
//!
//! # fn main() -> Result<()> {
//! let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
//! let seed = 123;
//! let alignment_length = 20;
//! let simulator = SubstitutionSimulator::new(
//!     SubstModel::<GTR>::new(&[0.25; 4], &[1.0; 6]),
//!     tree,
//!     DefaultGenerator::new(seed),
//!     alignment_length,
//! )?;
//! let msa: MASA = simulator.simulate_ancestral_alignment();
//! assert_eq!(msa.seq_count(), 3);
//! assert_eq!(msa.len(), alignment_length);
//! # Ok(()) }
//! ```

use std::cell::RefCell;

use bio::io::fasta::Record;
use hashbrown::HashMap;
use log::warn;
use rand::{distr::weighted::WeightedIndex, Rng, RngCore, SeedableRng};

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::alphabets::Alphabet;
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::random::RandomGenerator;
use crate::record_wo_desc as record;
use crate::substitution_models::{QMatrix, SubstModel};
use crate::tree::{NodeIdx, Tree};
use crate::{Result, MAX_BLEN, REPORT_ISSUES_URL};

/// Simulates substitutions on a fixed-length ancestral alignment.
///
/// The root sequence is sampled from the substitution model's stationary distribution and
/// evolved down the tree along the branches. The simulation produces no gaps; it is useful
/// on its own or in combination with an indel simulator, where the indel-only MSA is used as
/// a mask over this substitution-only MSA (e.g. [`crate::tkf_model::simulate_msa::TKFMSASimulator`]).
#[derive(Debug, Clone)]
pub struct SubstitutionSimulator<R>
where
    R: Rng + SeedableRng + RngCore,
{
    tree: Tree,
    alphabet: Alphabet,
    root_dist: WeightedIndex<f64>,
    /// Probability distributions for each branch (NodeIdx) and each parent character (index in the Vec).
    p_weighted: HashMap<NodeIdx, Vec<WeightedIndex<f64>>>,
    rng: RefCell<RandomGenerator<R>>,
    alignment_length: usize,
}

fn unwrap_weighted_index(slice: &[f64]) -> WeightedIndex<f64> {
    WeightedIndex::new(slice).unwrap_or_else(|e| {
        panic!(
            "Getting WeightedIndex from slice failed. This should never happen. \
            Please report this at {REPORT_ISSUES_URL}. \
            Error: {e}"
        )
    })
}

impl<R> SubstitutionSimulator<R>
where
    R: Rng + SeedableRng + RngCore,
{
    /// Create a new SubstitutionSimulator with the given substitution model, tree, RNG and
    /// alignment length.
    ///
    /// A zero `alignment_length` is accepted, but simulation will emit a warning and return an empty alignment.
    ///
    /// # Errors
    /// * If the tree node IDs cannot be completed (see
    ///   [`set_missing_tree_node_ids`](`crate::phylo_info::set_missing_tree_node_ids`)).
    pub fn new<Q: QMatrix>(
        model: SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
        alignment_length: usize,
    ) -> Result<Self> {
        let tree = set_missing_tree_node_ids(&tree)?;
        let root_dist = unwrap_weighted_index(model.qmatrix.freqs().as_slice());

        let mut p_weighted = HashMap::with_capacity(tree.len());
        for idx in tree.preorder().iter().skip(1) {
            let blen = tree.node(idx).blen;
            let qmat = model.qmatrix.q();
            let p = if blen > MAX_BLEN {
                (qmat * MAX_BLEN).exp()
            } else {
                (qmat * blen).exp()
            };

            let mut column_dists = Vec::with_capacity(p.ncols());
            for col in 0..p.ncols() {
                let column = p.column(col);
                let weighted_index = unwrap_weighted_index(column.as_slice());
                column_dists.push(weighted_index);
            }
            p_weighted.insert(*idx, column_dists);
        }

        let alphabet = *Q::alphabet();

        Ok(Self {
            tree,
            alphabet,
            root_dist,
            p_weighted,
            rng: RefCell::new(rng),
            alignment_length,
        })
    }

    /// Sets the alignment length for the simulation.
    ///
    /// A zero `alignment_length` is accepted, but simulation will emit a warning and return an empty alignment.
    pub fn alignment_length(&mut self, length: usize) {
        self.alignment_length = length;
    }

    fn simulate_root_sequence(&self, alignment_length: usize) -> Vec<usize> {
        let mut rng = self.rng.borrow_mut();
        let root_seq: Vec<usize> = (0..alignment_length)
            .map(|_| rng.sample(&self.root_dist))
            .collect();
        drop(rng);
        root_seq
    }

    fn simulate_child_sequence(&self, parent_seq: &[usize], node_idx: NodeIdx) -> Vec<usize> {
        let column_dists = self.p_weighted.get(&node_idx).unwrap();
        let mut rng = self.rng.borrow_mut();
        let child_seq: Vec<usize> = parent_seq
            .iter()
            .map(|&parent_state| rng.sample(&column_dists[parent_state]))
            .collect();
        drop(rng);
        child_seq
    }

    fn sequences_to_records(&self, sequences: &HashMap<NodeIdx, Vec<usize>>) -> Vec<Record> {
        sequences
            .iter()
            .map(|(node_idx, seq)| {
                let id = self.tree.node_id(node_idx);
                let char_seq: Vec<u8> = seq.iter().map(|&s| self.alphabet.symbols()[s]).collect();
                record!(id, &char_seq)
            })
            .collect()
    }

    fn simulate_sequences(&self, alignment_length: usize) -> Sequences {
        let mut sequences = HashMap::with_capacity(self.tree.len());

        sequences.insert(
            self.tree.root,
            self.simulate_root_sequence(alignment_length),
        );

        for node_idx in self.tree.preorder().iter().skip(1) {
            let parent_idx = self.tree.parent(node_idx).unwrap();
            let parent_seq = sequences.get(&parent_idx).unwrap();
            sequences.insert(
                *node_idx,
                self.simulate_child_sequence(parent_seq, *node_idx),
            );
        }

        let records = self.sequences_to_records(&sequences);

        Sequences::new(records).unwrap_or_else(|e| {
            panic!(
                "Creating Sequences from simulated records failed. This should never happen. \
                Please report this at {REPORT_ISSUES_URL}. \
                Error: {e}"
            )
        })
    }

    pub(crate) fn simulate_ancestral_alignment_with_length<AA: AncestralAlignment>(
        &self,
        alignment_length: usize,
    ) -> PhyloInfo<AA> {
        if alignment_length == 0 {
            warn!("Alignment length is set to 0, will produce an empty alignment");
        }

        let seqs = self.simulate_sequences(alignment_length);

        let msa = AA::from_aligned_with_ancestral(seqs, &self.tree).unwrap_or_else(|e| {
            panic!(
                "Creating AncestralAlignment from simulated Sequences failed. This should never happen. \
                Please report this at {REPORT_ISSUES_URL}. \
                Error: {e}"
            )
        });
        PhyloInfo { msa, tree: self.tree.clone()}
    }
}

impl<R> AlignmentSimulation for SubstitutionSimulator<R>
where
    R: Rng + SeedableRng + RngCore,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> PhyloInfo<AA> {
        self.simulate_ancestral_alignment_with_length(self.alignment_length)
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {

    use crate::alignment::{Alignment, MASA};
    use crate::random::{DefaultGenerator, FakeGenerator};
    use crate::substitution_models::{dna_models::GTR, SubstModel};
    use crate::tree;

    use super::*;

    #[test]
    fn test_substitution_simulator() {
        // GTR with chosen freqs and rate parameters
        let model = SubstModel::<GTR>::new(&[0.3, 0.2, 0.2, 0.3], &[0.8, 1.2, 0.9, 1.1, 0.7]);
        let tree = tree!("((A:2.0,B:2.0)AB:2.0,(C:2.0,D:2.0)CD:2.0)R;");
        let mut rng = FakeGenerator::default();
        for _ in 0..10 {
            print!("{}", rng.random::<u32>());
        }

        let simulator = SubstitutionSimulator::new(model, tree.clone(), rng, 50).unwrap();

        let alignment: MASA = simulator.simulate_ancestral_alignment().msa;

        assert_eq!(alignment.len(), 50);
        assert_eq!(
            alignment.seq_count() + alignment.ancestral_seqs().len(),
            tree.len()
        );
        // no gaps in this simulation, so all sequences should have the same length as the alignment
        for seq in alignment.seqs() {
            assert_eq!(seq.seq().len(), alignment.len());
        }
    }

    #[test]
    fn test_reproducibility() {
        // GTR with same parameters to ensure reproducibility across RNGs
        let model = SubstModel::<GTR>::new(&[0.3, 0.2, 0.2, 0.3], &[0.8, 1.2, 0.9, 1.1, 0.7]);
        let tree = tree!("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;");

        let rng1 = DefaultGenerator::new(42);
        let rng2 = DefaultGenerator::new(42);

        let simulator1 =
            SubstitutionSimulator::new(model.clone(), tree.clone(), rng1, 100).unwrap();
        let simulator2 = SubstitutionSimulator::new(model, tree.clone(), rng2, 100).unwrap();

        let alignment1: MASA = simulator1.simulate_ancestral_alignment().msa;
        let alignment2: MASA = simulator2.simulate_ancestral_alignment().msa;

        assert_eq!(
            alignment1.to_string(),
            alignment2.to_string(),
            "Same seed should produce identical alignments"
        );
    }
}
