use std::fmt::Display;

use itertools::any;
use num_enum::{FromPrimitive, IntoPrimitive};
use rand::{Rng, SeedableRng};

use crate::alignment::AncestralAlignment;
use crate::likelihood::ParamRange;
use crate::random::RandomGenerator;
use crate::tkf_model::simulate_msa::{ExpectedRootLength, FragmentSampler};
use crate::tkf_model::{TKFModel, DEFAULT_LAMBDA, DEFAULT_MU};
use crate::{bail, Result};

pub const TKF91_PARAM_COUNT: usize = 2;

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF91Parameters {
    Lambda = 0,
    Mu = 1,
    #[num_enum(catch_all)]
    Invalid(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF91IndelModel {
    params: Vec<f64>,
}

impl TKF91IndelModel {
    /// Creates a new TKF91 model given parameters: lambda, mu.
    /// Validates the parameters and returns an error if they are invalid:
    /// 0 < lambda < mu < infinity.
    pub fn new(params: &[f64]) -> Result<Self> {
        if params.len() != TKF91_PARAM_COUNT {
            bail!(
                Tkf,
                "Expected {} parameters for TKF91 model (lambda, mu), but got {}",
                TKF91_PARAM_COUNT,
                params.len()
            );
        }

        if any(params, |&p| p.is_nan() || p.is_infinite() || p <= 0.0) {
            bail!(
                Tkf,
                "All parameters for TKF91 model must be positive non-nan and not infinite, but got {:?}",
                params
            );
        }

        if params[usize::from(TKF91Parameters::Lambda)] >= params[usize::from(TKF91Parameters::Mu)]
        {
            bail!(
                Tkf,
                "Parameter lambda must be less than mu, but got lambda = {} and mu = {}",
                params[usize::from(TKF91Parameters::Lambda)],
                params[usize::from(TKF91Parameters::Mu)]
            );
        }

        Ok(Self {
            params: params.to_vec(),
        })
    }
}

impl Default for TKF91IndelModel {
    fn default() -> Self {
        Self {
            params: vec![DEFAULT_LAMBDA, DEFAULT_MU],
        }
    }
}

impl TKFModel for TKF91IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF91Parameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF91Parameters::Mu)]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        self.params[idx] = value;
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let param = TKF91Parameters::from_primitive(idx);
        match param {
            TKF91Parameters::Lambda => (f64::EPSILON, self.mu() - f64::EPSILON),
            TKF91Parameters::Mu => (self.lambda() + f64::EPSILON, f64::MAX),
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
    }

    fn ln_insertion_factor_at_root(&self) -> f64 {
        // TODO: this lambda.ln() and mu.ln() could be cached, see issue #152 https://github.com/acg-team/phyne/issues/152
        self.lambda().ln() - self.mu().ln()
    }

    fn ln_insertion_factor_at_non_root(&self, ln_beta: f64) -> f64 {
        self.lambda().ln() + ln_beta
    }

    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64 {
        (block_len as f64) * ln_tree_event_factor
    }

    /// Since TKF91 is a single-residue indel model, each position is its own block.
    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        (1..msa.len() + 1).collect()
    }
}

impl Display for TKF91IndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF91 with lambda = {}, mu = {}",
            self.lambda(),
            self.mu(),
        )
    }
}

impl FragmentSampler for TKF91IndelModel {
    /// In TKF91 every residue is its own independent link, so the fragment
    /// length is always 1 and its log-probability is ln(1) = 0.0.
    fn sample_fragment_length<R: Rng + SeedableRng>(
        &self,
        _rng: &mut RandomGenerator<R>,
    ) -> (usize, f64) {
        (1, 0.0)
    }
}

impl ExpectedRootLength for TKF91IndelModel {
    fn expected_root_length(&self) -> f64 {
        (self.lambda() / self.mu()) / (1.0 - (self.lambda() / self.mu()))
    }
}

#[cfg(test)]
mod private_tests {
    use super::*;
    use crate::Error;
    use assert_matches::assert_matches;
    use rstest::rstest;

    #[test]
    fn tkf91_indel_model_new_success() {
        let model = TKF91IndelModel::new(&[0.5, 1.0]).unwrap();
        assert_eq!(model.lambda(), 0.5);
        assert_eq!(model.mu(), 1.0);
        assert_eq!(model.params(), &[0.5, 1.0]);
    }

    #[rstest]
    #[case::too_few(&[0.5])]
    #[case::too_many(&[0.5, 1.0, 2.0])]
    fn tkf91_indel_model_new_wrong_param_count(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("Expected 2 parameters for TKF91 model")
        );
    }

    #[rstest]
    #[case::zero_lambda(&[0.0, 1.0])]
    #[case::zero_mu(&[0.5, 0.0])]
    #[case::negative_lambda(&[-0.5, 1.0])]
    #[case::nan_lambda(&[f64::NAN, 1.0])]
    #[case::infinite_mu(&[0.5, f64::INFINITY])]
    fn tkf91_indel_model_new_non_positive_or_non_finite(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("must be positive non-nan and not infinite")
        );
    }

    #[rstest]
    #[case::lambda_eq_mu(&[1.0, 1.0])]
    #[case::lambda_gt_mu(&[2.0, 1.0])]
    fn tkf91_indel_model_new_lambda_ge_mu(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("Parameter lambda must be less than mu")
        );
    }

    #[test]
    #[should_panic]
    fn tkf91_param_range_invalid_index() {
        let model = TKF91IndelModel {
            params: vec![0.5, 1.0],
        };
        // Use an invalid index
        model.param_range(2);
    }

    #[test]
    fn tkf91_model_fmt() {
        let tkf_indel_model = TKF91IndelModel {
            params: vec![1.1, 2.0],
        };

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(fmt, "TKF91 with lambda = 1.1, mu = 2");
    }

    #[test]
    fn tkf91_indel_set_param() {
        let mut model = TKF91IndelModel {
            params: vec![1.0, 2.0],
        };
        let new_lambda = 1.1;
        model.set_param(usize::from(TKF91Parameters::Lambda), new_lambda);
        assert_eq!(model.lambda(), new_lambda);
        let new_mu = 2.1;
        model.set_param(usize::from(TKF91Parameters::Mu), new_mu);
        assert_eq!(model.mu(), new_mu);
    }
}
