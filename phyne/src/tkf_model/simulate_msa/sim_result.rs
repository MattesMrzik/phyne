use crate::alignment::{surviving_columns_mask, AncestralAlignment};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::simulate_msa::Fragmentation;

/// Result of a TKF simulation (indels only, or indels + substitutions) including fragmentation.
pub struct TKFSimulationResult<AA: AncestralAlignment> {
    /// The simulated PhyloInfo containing the ancestral alignment and the tree.
    pub phylo: PhyloInfo<AA>,
    /// The fragmentation (right-exclusive boundaries of fragments) in sync with `masa`.
    pub fragmentation: Fragmentation,
}

impl<AA: AncestralAlignment> TKFSimulationResult<AA> {
    /// Removes columns where the ancestral character goes extinct (i.e., all leaf sequences
    /// have a gap) from the alignment and keeps the fragmentation in sync.
    ///
    /// # Panics
    /// * If the fragmentation does not match the alignment, e.g. when the mask would split a
    ///   fragment. This indicates a critical failure/bug.
    pub fn remove_extinct_columns(&mut self) {
        let keep_col_mask = surviving_columns_mask(&self.phylo.msa);
        self.phylo.msa.remove_extinct_columns();
        self.fragmentation.remove_cols(&keep_col_mask);
        self.fragmentation
            .fragmentation_works_with_ancestral_alignment(&self.phylo.msa);
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use crate::alignment::{Alignment, Sequences, MASA};
    use crate::tree::Tree;
    use crate::{record_wo_desc, tree, Record};

    use super::*;

    /// Builds a [`TKFSimulationResult`] from raw records. The fragmentation is chosen to have
    /// one fragment per column (`[1, 2, ..., n]`), which is always consistent with the alignment
    /// and lets us check that [`TKFSimulationResult::remove_extinct_columns`] keeps it in sync.
    fn result_from(
        records: Vec<Record>,
        tree: Tree,
        fragmentation: Fragmentation,
    ) -> TKFSimulationResult<MASA> {
        let phylo =
            MASA::from_aligned_with_ancestral(Sequences::new(records).unwrap(), tree).unwrap();
        TKFSimulationResult {
            phylo,
            fragmentation,
        }
    }

    #[test]
    fn remove_extinct_columns_no_extinction() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
        let fragmentation = Fragmentation::new(vec![1, 3, 4]);
        let mut result = result_from(
            vec![
                record_wo_desc!("A", b"ACGT"),
                record_wo_desc!("B", b"ACGT"),
                record_wo_desc!("C", b"ACGT"),
                record_wo_desc!("I", b"ACGT"),
                record_wo_desc!("R", b"ACGT"),
            ],
            tree,
            fragmentation,
        );

        // no column goes extinct in the leaves, so nothing is removed
        assert_eq!(result.phylo.msa.len(), 4);
        result.remove_extinct_columns();
        assert_eq!(result.phylo.msa.len(), 4);
        assert_eq!(
            result.fragmentation.right_exclusive_boundaries(),
            [1, 2, 3, 4].as_slice()
        );
    }

    #[test]
    fn remove_extinct_columns_keeps_fragmentation_in_sync() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
        let fragmentation = Fragmentation::new(vec![1, 2, 5, 6, 7, 9]);
        let mut result = result_from(
            vec![
                record_wo_desc!("A", b"A-TTT----"),
                record_wo_desc!("B", b"C-CTT----"),
                record_wo_desc!("C", b"G-CTT--GG"),
                record_wo_desc!("I", b"T-CTTAAGG"),
                record_wo_desc!("R", b"AAATTTGGG"),
            ],
            tree,
            fragmentation,
        );

        result.remove_extinct_columns();

        assert_eq!(result.phylo.msa.len(), 6);
        // the fragmented columns are dropped together with the extinct columns
        assert_eq!(
            result.fragmentation.right_exclusive_boundaries(),
            [1, 4, 6].as_slice()
        );
    }

    #[test]
    fn remove_extinct_columns_all_extinct() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
        let fragmentation = Fragmentation::new(vec![4, 5]);
        let mut result = result_from(
            vec![
                record_wo_desc!("A", b"----"),
                record_wo_desc!("B", b"----"),
                record_wo_desc!("C", b"----"),
                record_wo_desc!("I", b"ACGT"),
                record_wo_desc!("R", b"ACGT"),
            ],
            tree,
            fragmentation,
        );

        result.remove_extinct_columns();

        assert_eq!(result.phylo.msa.len(), 0);
        assert!(result.fragmentation.is_empty());
    }
}
