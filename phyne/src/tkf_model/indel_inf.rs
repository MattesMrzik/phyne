use std::cell::RefCell;
use std::num::NonZeroUsize;

use log::{debug, info};
use rand::{Rng, RngCore, SeedableRng};
use rand_pcg::Pcg32;

use crate::alignment::{Alignment, AncestralAlignment};
use crate::asr::AncestralSequenceReconstruction;
use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::{ModelOptimiser, StopCondition};
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::random::{DefaultGenerator, RandomGenerator};
use crate::tkf_model::{EdgeSeqsReestimator, TKFIndelCost, TKFIndelCostBuilder, TKFModel};
use crate::tree::NodeIdx;
use crate::{bail, Result, DEFAULT_EPSILON};

/// Infers the homology path (the indel points, i.e. the presence/absence of characters)
/// at the ancestral nodes of a [tree](`crate::tree::Tree`) under a [TKF](`TKFModel`) indel model.
///
/// Starting from an initial guess of the indel points, it iteratively re-estimates the
/// ancestral gap/non-gap pattern (see [`EdgeSeqsReestimator`]) until convergence and,
/// optionally, optimises the indel model parameters. The result is a [`PhyloInfo`] whose
/// [ancestral alignment](`AncestralAlignment`) has ancestral characters that are wild cards
/// ([`AMB_CHAR`](`crate::alphabets::AMB_CHAR`)), encoding the homology path.
///
/// # Example
/// ```rust
/// use phyne::alignment::{Alignment, AncestralAlignment, MASA, MSA, Sequences};
/// use phyne::phylo_info::PhyloInfo;
/// use phyne::tkf_model::{TKF92IndelModel, TKFIndelInf};
/// use phyne::{record_wo_desc as record, tree};
///
/// # use phyne::Result;
/// # fn main() -> Result<()> {
/// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
/// let phylo = MSA::from_aligned(
///     Sequences::new(vec![
///         record!("A", b"ACGT"),
///         record!("B", b"ACGT"),
///         record!("C", b"ACGT"),
///     ])?,
///     tree,
/// )?;
/// let inf = TKFIndelInf::<TKF92IndelModel>::default();
/// let (ancestral_phylo, _model) = inf.estimate_homology_path::<_, MASA>(phylo)?;
/// assert_eq!(ancestral_phylo.msa.seqs().len(), 3);
/// assert_eq!(ancestral_phylo.msa.ancestral_seqs().len(), 2);
/// # Ok(()) }
/// ```
pub struct TKFIndelInf<T, R = Pcg32>
where
    T: TKFModel,
    R: Rng + SeedableRng + RngCore,
{
    epsilon: f64,
    max_iterations: Option<usize>,
    param_opt: bool,
    indel_model: T,
    // rng for reproducibility when breaking ties in the re-estimation process
    rng: RefCell<RandomGenerator<R>>,
}

impl<T: TKFModel> Default for TKFIndelInf<T, Pcg32> {
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            max_iterations: None,
            param_opt: false,
            indel_model: T::default(),
            rng: RefCell::new(DefaultGenerator::default()),
        }
    }
}

impl<T, R> TKFIndelInf<T, R>
where
    T: TKFModel,
    R: Rng + SeedableRng + RngCore,
{
    // TODO: what other ctors do we want?, perhaps some with default model but passing the other params
    pub fn new_with_attr(
        indel_model: T,
        epsilon: f64,
        max_iterations: Option<usize>,
        param_opt: bool,
        rng: RandomGenerator<R>,
    ) -> Result<Self> {
        if epsilon < 0.0 {
            bail!(
                Tree,
                "epsilon must be greater than or equal to 0, but got {}",
                epsilon
            );
        }

        if let Some(max_iters) = max_iterations {
            if max_iters == 0 {
                bail!(
                    Tree,
                    "max_iterations must be greater than 0, but got {}",
                    max_iters
                );
            }
        }
        Ok(Self {
            epsilon,
            max_iterations,
            param_opt,
            indel_model,
            rng: RefCell::new(rng),
        })
    }
}

impl<T, R, A, AA> AncestralSequenceReconstruction<A, AA> for TKFIndelInf<T, R>
where
    T: TKFModel,
    R: Rng + SeedableRng + RngCore,
    A: Alignment,
    AA: AncestralAlignment,
{
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA> {
        self.estimate_homology_paths(phylo).unwrap().phylo
    }

    fn reconstruct_ancestral_seqs(&self, mut phylo: PhyloInfo<A>) -> Result<PhyloInfo<AA>> {
        set_missing_tree_node_ids(&mut phylo.tree)?;
        let result = self.estimate_homology_paths(phylo)?;
        Ok(result.phylo)
    }
}

/// Result of the [`TKFIndelInf::estimate_homology_paths`] and [`TKFIndelInf::estimate_homology_paths_from_initial`] methods.
/// It contains the [`PhyloInfo`] with the inferred homology paths, and the [`TKFModel`] that was
/// used and optionally optimised during the inference process.
pub struct TKFIndelInfResult<T, AA>
where
    T: TKFModel,
    AA: AncestralAlignment,
{
    pub phylo: PhyloInfo<AA>,
    pub model: T,
}

impl<T: TKFModel, R: Rng + SeedableRng + RngCore> TKFIndelInf<T, R> {
    pub fn estimate_homology_paths<A: Alignment, AA: AncestralAlignment>(
        &self,
        phylo: PhyloInfo<A>,
    ) -> Result<TKFIndelInfResult<T, AA>> {
        let asr = ParsimonyPresenceAbsence {};
        let initial_indel_points = asr.reconstruct_ancestral_seqs(phylo)?;
        self.estimate_homology_paths_from_initial(initial_indel_points)
    }

    pub fn estimate_homology_paths_from_initial<AA: AncestralAlignment>(
        &self,
        phylo: PhyloInfo<AA>,
    ) -> Result<TKFIndelInfResult<T, AA>> {
        let max_iters = NonZeroUsize::new(self.max_iterations.unwrap_or(usize::MAX)).unwrap();
        let stop_condition = StopCondition::max_iter_epsilon(max_iters, self.epsilon);
        let intermediate_stop_condition = StopCondition::epsilon(self.epsilon);

        let mut cost = TKFIndelCostBuilder::new(self.indel_model.clone(), phylo).build();
        let mut prev_cost = f64::NEG_INFINITY;
        let mut curr_cost = cost.cost();
        let mut delta = curr_cost - prev_cost;

        let mut iterations = 0;
        while stop_condition.should_continue(iterations, delta) {
            iterations += 1;
            info!("Iteration: {iterations}, current cost: {curr_cost}");
            prev_cost = curr_cost;
            if self.param_opt {
                cost = Self::optimise_params(cost, intermediate_stop_condition)?;
            }
            curr_cost = Self::reestimate_edges(&mut cost, &mut self.rng.borrow_mut(), curr_cost);
            delta = curr_cost - prev_cost;
        }

        Ok(TKFIndelInfResult {
            phylo: cost.phylo,
            model: cost.model,
        })
    }

    /// Optimises the model parameters of the given cost.
    fn optimise_params<AA: AncestralAlignment + Clone>(
        cost: TKFIndelCost<T, AA>,
        intermediate_stop_condition: StopCondition,
    ) -> Result<TKFIndelCost<T, AA>> {
        let model_optimiser = ModelOptimiser::with_stop_condition(
            cost,
            FrequencyOptimisation::Fixed, // as these don't influence the indel points
            intermediate_stop_condition,
        );
        let mode_opti_result = model_optimiser.run()?;
        Ok(mode_opti_result.cost)
    }

    /// Re-estimates the ancestral sequences at all non-root internal nodes of the tree,
    /// keeping all other sequences and the tree fixed.
    /// Returns the resulting cost after re-estimation.
    fn reestimate_edges<AA: AncestralAlignment>(
        cost: &mut TKFIndelCost<T, AA>,
        rng: &mut RandomGenerator<R>,
        mut curr_cost: f64,
    ) -> f64 {
        let mut reestimator = EdgeSeqsReestimator::new(cost, rng);
        for node in reestimator.phylo().tree.postorder().clone() {
            if node == reestimator.phylo().tree.root {
                continue;
            }
            match node {
                NodeIdx::Internal(_) => {
                    let prev_cost = curr_cost;
                    curr_cost = reestimator.reestimate_unchecked(&node);
                    debug!(
                        "Re-estimated sequences at node {node}, cost after re-estimation: {curr_cost}",
                    );
                    if curr_cost > prev_cost {
                        debug!("Cost improved");
                    }
                }
                NodeIdx::Leaf(_) => {}
            }
        }
        curr_cost
    }
}
