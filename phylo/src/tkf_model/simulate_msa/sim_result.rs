use crate::alignment::{surviving_columns_mask, AncestralAlignment};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::simulate_msa::Fragmentation;
use crate::Result;

/// Result of a TKF simulation (indels only, or indels + substitutions) including
/// fragmentation and the tree the simulation was performed on.
pub struct TKFSimulationResult<AA: AncestralAlignment> {
    // TODO: have phylo instead.
    // TODO: prune_empty_leaves on phylo
    /// The simulated ancestral alignment.
    pub phylo: PhyloInfo<AA>,
    /// The fragmentation (right-exclusive boundaries of fragments) in sync with `masa`.
    pub fragmentation: Fragmentation,
}

impl<AA: AncestralAlignment> TKFSimulationResult<AA> {
    /// Removes columns where the ancestral character goes extinct (i.e., all leaf sequences
    /// have a gap) from the alignment and keeps the fragmentation in sync.
    ///
    /// # Errors
    /// * If the fragmentation does not match the alignment, e.g. when the mask would split a
    ///   fragment.
    pub fn remove_extinct_columns(&mut self) -> Result<()> {
        let keep_col_mask = surviving_columns_mask(&self.phylo.msa);
        self.phylo.msa.remove_extinct_columns();
        self.fragmentation.remove_cols(&keep_col_mask)?;
        self.fragmentation
            .fragmentation_works_with_ancestral_alignment(&self.phylo.msa)
    }
}


