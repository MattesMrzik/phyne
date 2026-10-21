use std::cell::RefCell;
use std::num::NonZeroUsize;

use log::info;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;

use crate::alignment::{Alignment, AncestralAlignment};
use crate::asr::AncestralSequenceReconstruction;
use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::{ModelOptimiser, StopCondition};
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::PhyloInfo;
use crate::random::{DefaultGenerator, RandomGenerator};
use crate::tkf_model::{EdgeSeqsReestimator, TKFIndelCost, TKFIndelCostBuilder, TKFModel};
use crate::tree::NodeIdx;
use crate::{bail, Result, DEFAULT_EPSILON};

/// Options for whether to optimise the [`indel model`](TKFModel) parameters during the
/// [`TKFIndelInf`] inference process.
pub enum IndelInfParamOpt {
    Enabled,
    Disabled,
}

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
/// let result = inf.estimate_homology_paths::<_, MASA>(phylo)?;
/// assert_eq!(result.phylo.msa.seqs().len(), 3);
/// assert_eq!(result.phylo.msa.ancestral_seqs().len(), 2);
/// # Ok(()) }
/// ```
pub struct TKFIndelInf<T, R = Pcg32>
where
    T: TKFModel,
    R: Rng + SeedableRng,
{
    epsilon: f64,
    max_iterations: Option<usize>,
    param_opt: IndelInfParamOpt,
    indel_model: T,
    /// It is wrapped in a RefCell to allow interior mutability, as the re-estimation process needs to mutate
    /// the rng state and the methods of [`AncestralSequenceReconstruction`] require &self, not &mut self.
    rng: RefCell<RandomGenerator<R>>,
}

impl<T: TKFModel + Default> Default for TKFIndelInf<T, Pcg32> {
    /// Creates a new [`TKFIndelInf`] with default attributes.
    ///
    /// It uses the default [`TKFModel`] implementation for the indel model, a default epsilon of
    /// [`DEFAULT_EPSILON`], no maximum number of iterations, and no parameter optimisation.
    /// The `rng` is seeded with a random seed and is used for breaking ties in the re-estimation
    /// dynamic programming process. It advances across iterations, so the results are not
    /// reproducible across different runs of the same code. For reproducible results, use
    /// [`TKFIndelInf::new_with_attr`] and provide your own `rng`.
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            max_iterations: None,
            param_opt: IndelInfParamOpt::Disabled,
            indel_model: T::default(),
            rng: RefCell::new(DefaultGenerator::default()),
        }
    }
}

impl<T, R> TKFIndelInf<T, R>
where
    T: TKFModel,
    R: Rng + SeedableRng,
{
    /// Creates a new [`TKFIndelInf`] with the given attributes.
    ///
    /// The `rng` is for reproducibility when breaking ties in the re-estimation dynamic programming process.
    pub fn new_with_attr(
        indel_model: T,
        epsilon: f64,
        max_iterations: Option<usize>,
        param_opt: IndelInfParamOpt,
        rng: RandomGenerator<R>,
    ) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!(Asr, "epsilon must be greater than 0, but got {}", epsilon);
        }

        if let Some(max_iters) = max_iterations {
            if max_iters == 0 {
                bail!(
                    Asr,
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
    R: Rng + SeedableRng,
    A: Alignment,
    AA: AncestralAlignment,
{
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA> {
        self.estimate_homology_paths(phylo).unwrap().phylo
    }

    fn reconstruct_ancestral_seqs(&self, phylo: PhyloInfo<A>) -> Result<PhyloInfo<AA>> {
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

impl<T: TKFModel, R: Rng + SeedableRng> TKFIndelInf<T, R> {
    /// Estimates the homology paths (ancestral gap/non-gap patterns) starting from a parsimony
    /// reconstruction of the ancestral sequences at the internal nodes of the tree.
    /// It then iteratively re-estimates the ancestral sequences at the internal nodes of the tree
    /// until convergence and, optionally ([`TKFIndelInf::estimate_homology_paths_from_initial`]).
    pub fn estimate_homology_paths<A: Alignment, AA: AncestralAlignment>(
        &self,
        phylo: PhyloInfo<A>,
    ) -> Result<TKFIndelInfResult<T, AA>> {
        let asr = ParsimonyPresenceAbsence {};
        // set missing tree node ids is called by the reconstruct_ancestral_seqs
        let initial_indel_points = asr.reconstruct_ancestral_seqs(phylo)?;
        self.estimate_homology_paths_from_initial(initial_indel_points)
    }

    /// Estimates the homology paths (ancestral gap/non-gap patterns) starting from an initial guess
    /// of the ancestral sequences at the internal nodes of the tree.
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
            if let IndelInfParamOpt::Enabled = self.param_opt {
                cost = Self::optimise_params(cost, intermediate_stop_condition)?;
            }
            Self::reestimate_edges(&mut cost, &mut self.rng.borrow_mut());
            curr_cost = cost.cost();
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
    fn reestimate_edges<AA: AncestralAlignment>(
        cost: &mut TKFIndelCost<T, AA>,
        rng: &mut RandomGenerator<R>,
    ) {
        let mut reestimator = EdgeSeqsReestimator::new(cost, rng);
        for node in reestimator.phylo().tree.postorder().clone() {
            if node == reestimator.phylo().tree.root {
                continue;
            }
            if let NodeIdx::Internal(_) = node {
                reestimator.reestimate_unchecked(&node);
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use assert_matches::assert_matches;

    use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
    use crate::asr::AncestralSequenceReconstruction;
    use crate::likelihood::ModelSearchCost;
    use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
    use crate::random::DefaultGenerator;
    use crate::tkf_model::{TKF92IndelModel, TKFIndelCostBuilder, TKFModel};
    use crate::{record_wo_desc as record, tree, Error, DEFAULT_EPSILON};

    use super::*;

    const MAX_ITERATIONS: Option<usize> = Some(10);

    #[test]
    fn new_with_attr_fails_for_invalid_attributes() {
        let err = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            TKF92IndelModel::default(),
            -0.1,
            MAX_ITERATIONS,
            IndelInfParamOpt::Disabled,
            DefaultGenerator::new(123),
        )
        .err()
        .unwrap();
        assert_matches!(err, Error::Asr(msg)
            if msg.contains("epsilon must be greater than 0"));

        let err = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            TKF92IndelModel::default(),
            DEFAULT_EPSILON,
            Some(0),
            IndelInfParamOpt::Disabled,
            DefaultGenerator::new(123),
        )
        .err()
        .unwrap();
        assert_matches!(err, Error::Asr(msg)
            if msg.contains("max_iterations must be greater than 0"));
    }

    fn setup_msa() -> PhyloInfo<MSA> {
        let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
        MSA::from_aligned(
            Sequences::new(vec![
                record!("A1", b"--GTGGA---"),
                record!("B2", b"-------NNA"),
                record!("C4", b"AGG-------"),
            ])
            .unwrap(),
            tree,
        )
        .unwrap()
    }

    fn tkf92_logl(model: &TKF92IndelModel, phylo: &PhyloInfo<MASA>) -> f64 {
        ModelSearchCost::cost(&TKFIndelCostBuilder::new(model.clone(), phylo.clone()).build())
    }

    #[test]
    fn default_tkf_indel_inf_reconstructs_ancestral_seqs() {
        let inf = TKFIndelInf::<TKF92IndelModel>::default();

        let unchecked: PhyloInfo<MASA> = inf.reconstruct_ancestral_seqs_unchecked(setup_msa());
        let checked: PhyloInfo<MASA> = inf.reconstruct_ancestral_seqs(setup_msa()).unwrap();

        for phylo in [&unchecked, &checked] {
            assert_eq!(phylo.msa.seqs().len(), 3);
            assert_eq!(phylo.msa.ancestral_seqs().len(), 2);
            assert!(tkf92_logl(&TKF92IndelModel::default(), phylo).is_finite());
        }
    }

    #[test]
    fn estimate_homology_paths_without_param_optimisation() {
        let model = TKF92IndelModel::default();
        let inf = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            model.clone(),
            DEFAULT_EPSILON,
            MAX_ITERATIONS,
            IndelInfParamOpt::Disabled,
            DefaultGenerator::new(123),
        )
        .unwrap();

        let result = inf.estimate_homology_paths::<_, MASA>(setup_msa()).unwrap();

        result.phylo.check_dollos_constraint().unwrap();
        let logl = tkf92_logl(&result.model, &result.phylo);
        assert!(logl.is_finite());
        assert_eq!(result.model.params(), model.params());
    }

    #[test]
    fn estimate_homology_paths_with_param_optimisation() {
        let model = TKF92IndelModel::default();
        let inf = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            model.clone(),
            DEFAULT_EPSILON,
            MAX_ITERATIONS,
            IndelInfParamOpt::Enabled,
            DefaultGenerator::new(123),
        )
        .unwrap();

        let result = inf.estimate_homology_paths::<_, MASA>(setup_msa()).unwrap();

        result.phylo.check_dollos_constraint().unwrap();
        let logl = tkf92_logl(&result.model, &result.phylo);
        assert!(logl.is_finite());
        assert_ne!(result.model.params(), model.params());
    }

    #[test]
    fn param_optimisation_improves_likelihood() {
        let model = TKF92IndelModel::new(&[0.1, 0.2, 0.3]).unwrap();

        let inf_no_opt = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            model.clone(),
            DEFAULT_EPSILON,
            MAX_ITERATIONS,
            IndelInfParamOpt::Disabled,
            DefaultGenerator::new(123),
        )
        .unwrap();
        let result_no_opt = inf_no_opt
            .estimate_homology_paths::<_, MASA>(setup_msa())
            .unwrap();
        let logl_no_opt = tkf92_logl(&result_no_opt.model, &result_no_opt.phylo);

        let inf_opt = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            model,
            DEFAULT_EPSILON,
            MAX_ITERATIONS,
            IndelInfParamOpt::Enabled,
            DefaultGenerator::new(123),
        )
        .unwrap();
        let result_opt = inf_opt
            .estimate_homology_paths::<_, MASA>(setup_msa())
            .unwrap();
        let logl_opt = tkf92_logl(&result_opt.model, &result_opt.phylo);

        assert!(logl_no_opt.is_finite());
        assert!(logl_opt.is_finite());
        assert_ne!(result_opt.model.params(), result_no_opt.model.params());
        assert!(logl_opt > logl_no_opt);
    }

    /// Tree with a very short edge (`I6` to `D3`), such that parsimony places the insertions
    /// unique to `D3` on that edge, while under TKF it is much more likely to place them on the
    /// long edge above (`R8` to `I6`) and delete them on the edge to `C2`.
    fn setup_short_edge_msa() -> PhyloInfo<MSA> {
        let tree = tree!("((A0:1.0,B1:1.0)I5:1.0,(C2:1.0,D3:0.01)I6:1.0)R8:1.0;");
        MSA::from_aligned(
            Sequences::new(vec![
                record!("A0", b"AA----"),
                record!("B1", b"A-----"),
                record!("C2", b"---A--"),
                record!("D3", b"--AAAA"),
            ])
            .unwrap(),
            tree,
        )
        .unwrap()
    }

    #[test]
    fn estimate_homology_paths_from_initial_improves_logl_and_changes_phylo() {
        let model = TKF92IndelModel::default();
        let asr = ParsimonyPresenceAbsence {};
        let initial = asr
            .reconstruct_ancestral_seqs(setup_short_edge_msa())
            .unwrap();
        let initial_logl = tkf92_logl(&model, &initial);

        let inf = TKFIndelInf::<TKF92IndelModel>::new_with_attr(
            model,
            DEFAULT_EPSILON,
            MAX_ITERATIONS,
            IndelInfParamOpt::Disabled,
            DefaultGenerator::new(123),
        )
        .unwrap();
        let result = inf
            .estimate_homology_paths_from_initial(initial.clone())
            .unwrap();

        result.phylo.check_dollos_constraint().unwrap();
        let final_logl = tkf92_logl(&result.model, &result.phylo);
        assert!(final_logl > initial_logl);
        assert_ne!(
            initial.msa.ancestral_seqs(),
            result.phylo.msa.ancestral_seqs()
        );
    }
}
