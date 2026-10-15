use crate::alignment::{Alignment, AncestralAlignment};
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::Result;

/// Trait for ancestral sequence reconstruction.
pub trait AncestralSequenceReconstruction<A: Alignment, AA: AncestralAlignment> {
    /// Checks if all node IDs (leaves and internal) in the tree are unique and calls
    /// [`Self::reconstruct_ancestral_seqs_unchecked`].
    fn reconstruct_ancestral_seqs(&self, mut phylo: PhyloInfo<A>) -> Result<PhyloInfo<AA>> {
        set_missing_tree_node_ids(&mut phylo.tree)?;
        Ok(self.reconstruct_ancestral_seqs_unchecked(phylo))
    }

    /// Reconstructs ancestral sequences without any checks. Is called by
    /// [`Self::reconstruct_ancestral_seqs`].
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA>;
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
