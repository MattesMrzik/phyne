use std::num::NonZeroUsize;

use log::info;
use rand::{Rng, SeedableRng};

use crate::alignment::{Alignment, AncestralAlignment};
use crate::asr::AncestralSequenceReconstruction;
use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::{ModelOptimiser, StopCondition};
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::PhyloInfo;
use crate::random::{DefaultGenerator, RandomGenerator};
use crate::tkf_model::{EdgeSeqsReestimator, TKFIndelCost, TKFModel};
use crate::tree::{NodeIdx, Tree};
use crate::{bail, Result, DEFAULT_EPSILON};

pub struct TKFIndelASR<T: TKFModel> {
    epsilon: f64,
    max_iterations: Option<usize>,
    param_opt: bool,
    indel_model: T,
    seed: Option<u64>, // optional seed for reproducibility when breaking ties in the re-estimation process
}

impl<T: TKFModel> Default for TKFIndelASR<T> {
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            max_iterations: None,
            param_opt: false,
            indel_model: T::default(),
            seed: None,
        }
    }
}

impl<T: TKFModel> TKFIndelASR<T> {
    // what other ctors do we want?, perhaps some with default model but passing the other params
    pub fn new_with_attr(
        indel_model: T,
        epsilon: f64,
        max_iterations: Option<usize>,
        param_opt: bool,
        seed: Option<u64>,
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
            seed,
        })
    }
}

impl<T: TKFModel, A: Alignment, AA: AncestralAlignment> AncestralSequenceReconstruction<A, AA>
    for TKFIndelASR<T>
{
    fn reconstruct_ancestral_seqs_unchecked(
        &self,
        // this will take a phylo
        // will call add missing tree node ids, or perhaps only in in the checked version
        leaf_alignment: &A,
        tree: &Tree,
    ) -> AA {
        self.indel_asr(leaf_alignment, tree.clone()).unwrap().0
    }

    fn reconstruct_ancestral_seqs(&self, leaf_alignment: &A, tree: &Tree) -> Result<AA> {
        // TODO: set missing ids
        let result = self.indel_asr(leaf_alignment, tree.clone())?;
        Ok(result.0)
    }
}

impl<T: TKFModel> TKFIndelASR<T> {
    pub fn indel_asr<A: Alignment, AA: AncestralAlignment>(
        &self,
        leaf_alignment: &A,
        tree: Tree,
    ) -> Result<(AA, T)> {
        let asr = ParsimonyPresenceAbsence {};
        let initial_indel_points = asr.reconstruct_ancestral_seqs_unchecked(leaf_alignment, &tree);
        self.indel_asr_from_initial(initial_indel_points, tree)
    }

    // TODO: this should take a phylo info and also return it
    pub fn indel_asr_from_initial<AA: AncestralAlignment>(
        &self,
        initial_indel_points: AA,
        tree: Tree,
    ) -> Result<(AA, T)> {
        let mut rng = match self.seed {
            Some(seed) => DefaultGenerator::new(seed),
            None => DefaultGenerator::default(),
        };
        let max_iters = NonZeroUsize::new(self.max_iterations.unwrap_or(usize::MAX)).unwrap();
        let stop_condition = StopCondition::max_iter_epsilon(max_iters, self.epsilon);
        let intermediate_stop_condition = StopCondition::epsilon(self.epsilon);

        let phylo = PhyloInfo {
            tree,
            msa: initial_indel_points,
        };

        let mut cost = self.indel_model.indel_cost(phylo);
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
            curr_cost = Self::reestimate_edges(&mut cost, &mut rng, curr_cost);
            delta = curr_cost - prev_cost;
            // TODO: output also checkpointed msa and params
        }

        Ok((cost.phylo.msa, cost.model))
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
    fn reestimate_edges<AA: AncestralAlignment, R: Rng + SeedableRng>(
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
                    info!(
                        "Re-estimated sequences at node {node}, cost after re-estimation: {curr_cost}",
                    );
                    if curr_cost > prev_cost {
                        // TODO: if we iterate over the nodes and then find an edge that is changed, then call the reestimation again on neighboring nodes
                        info!("Cost improved");
                    }
                }
                NodeIdx::Leaf(_) => {}
            }
        }
        curr_cost
    }
}
