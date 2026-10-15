//! Simulation of ancestral alignments under a TKF model.
//!
//! This module provides simulators that evolve sequences down a [`Tree`](`crate::tree::Tree`)
//! according to the TKF indel model, optionally combined with a substitution model:
//!
//! - [`TKFIndelMSASimulator`] simulates only the indel process, producing an ancestral
//!   alignment of [`GAP`](`crate::alphabets::GAP`)s and [`AMB_CHAR`](`crate::alphabets::AMB_CHAR`)s
//!   that represents the indel history, together with a [`Fragmentation`].
//! - [`TKFMSASimulator`] runs the indel simulator first and then simulates substitutions
//!   on the resulting columns, placing gaps according to the indel MSA.
//!
//! Both simulators return a [`TKFSimulationResult`] containing the ancestral alignment
//! (e.g., an [`MASA`](`crate::alignment::MASA`)), the [`Fragmentation`] and the
//! [`Tree`](`crate::tree::Tree`). Note that some columns may go
//! extinct, i.e., all leaf sequences have a gap in that column, and that leaves whose
//! aligned sequence consists only of gaps may appear. Use
//! [`TKFSimulationResult::remove_extinct_columns`] to remove the extinct columns (and keep
//! the fragmentation in sync) and [`crate::phylo_info::PhyloInfo::prune_empty_leaves`] to prune the
//! gap-only leaves (and their ancestors) from both the tree and the alignment.
//!
//! # Example
//! ```
//! use phyne::alignment::{Alignment, MASA};
//! use phyne::random::DefaultGenerator;
//! use phyne::substitution_models::{dna_models::JC69, SubstModel};
//! use phyne::tkf_model::simulate_msa::TKFMSASimulator;
//! use phyne::tkf_model::TKF92IndelModel;
//! use phyne::tree;
//! # use phyne::Result;
//!
//! # fn main() -> Result<()> {
//! let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
//! let seed = 123;
//! // The maximum number of inserted links (=fragments) in a single insertion event.
//! let max_insertion_length = Some(50);
//! let simulator = TKFMSASimulator::new(
//!     TKF92IndelModel::default(),
//!     SubstModel::<JC69>::default(),
//!     tree,
//!     max_insertion_length,
//! )?;
//! let mut rng = DefaultGenerator::new(seed);
//! let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
//! assert_eq!(result.phylo.msa.seq_count(), 3);
//! assert!(!result.fragmentation.is_empty());
//! # Ok(()) }
//! ```

mod sim_result;
pub use sim_result::*;
mod sim_tkf_msa;
pub use sim_tkf_msa::*;
mod sim_tkf_indel_msa;
pub use sim_tkf_indel_msa::*;
mod fragmentation;
pub use fragmentation::*;
