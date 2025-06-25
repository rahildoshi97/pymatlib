import logging
from typing import Tuple, Union

import pwlf
import sympy as sp

from pymatlib.parsing.config.yaml_keys import REGRESSION_KEY, SIMPLIFY_KEY, DEGREE_KEY, SEGMENTS_KEY, EXTRAPOLATE_KEY, \
    CONSTANT_KEY
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)


class RegressionProcessor:
    """Handles all regression-related functionality."""

    @staticmethod
    def process_regression_params(prop_config: dict,
                                  prop_name: str,
                                  data_length: int) -> Tuple[
        bool, Union[str, None], Union[int, None], Union[int, None]]:
        """Process regression parameters from configuration."""
        has_regression = isinstance(prop_config, dict) and REGRESSION_KEY in prop_config
        if not has_regression:
            return False, None, None, None
        regression_config = prop_config[REGRESSION_KEY]
        simplify_type = regression_config[SIMPLIFY_KEY]
        degree = regression_config[DEGREE_KEY]
        segments = regression_config[SEGMENTS_KEY]
        if segments >= data_length:
            raise ValueError(f"Number of segments ({segments}) must be less than number of data points ({data_length})")
        if segments > ProcessingConstants.MAX_REGRESSION_SEGMENTS:
            raise ValueError(
                f"Number of segments ({segments}) is too high for {prop_name}. Please reduce it to {ProcessingConstants.MAX_REGRESSION_SEGMENTS} or less.")
        elif segments > ProcessingConstants.WARNING_REGRESSION_SEGMENTS:
            logger.warning(f"Number of segments ({segments}) for {prop_name} may lead to overfitting.")
        return has_regression, simplify_type, degree, segments

    @staticmethod
    def process_regression(temp_array, prop_array, T, lower_bound_type, upper_bound_type,
                           degree, segments, seed=ProcessingConstants.DEFAULT_REGRESSION_SEED):
        """Centralized regression processing logic."""
        v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=seed)
        v_pwlf.fit(n_segments=segments)
        return sp.Piecewise(*RegressionProcessor.get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))

    @staticmethod
    def get_symbolic_conditions(pwlf_: pwlf.PiecewiseLinFit, x: sp.Symbol, lower_: str, upper_: str):
        """Create symbolic conditions for a piecewise function from a pwlf fit."""
        conditions = []
        # Special case: single segment with extrapolation at both ends
        if pwlf_.n_segments == 1 and lower_ == EXTRAPOLATE_KEY and upper_ == EXTRAPOLATE_KEY:
            eqn = RegressionProcessor.get_symbolic_eqn(pwlf_, 1, x)
            conditions.append((eqn, x >= -sp.oo))
            return conditions
        # Handle lower bound for all cases
        if lower_ == CONSTANT_KEY:
            eqn = RegressionProcessor.get_symbolic_eqn(pwlf_, 1, x)
            conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[0]}), x < pwlf_.fit_breaks[0]))
        # Process all segments
        for i in range(pwlf_.n_segments):
            eqn = RegressionProcessor.get_symbolic_eqn(pwlf_, i + 1, x)
            # First segment with extrapolation
            if i == 0 and lower_ == EXTRAPOLATE_KEY:
                conditions.append((eqn, x < pwlf_.fit_breaks[i + 1]))
            # Last segment with extrapolation
            elif i == pwlf_.n_segments - 1 and upper_ == EXTRAPOLATE_KEY:
                conditions.append((eqn, x >= pwlf_.fit_breaks[i]))
            else:  # Regular intervals (including first/last with constant bounds)
                conditions.append((eqn, sp.And(x >= pwlf_.fit_breaks[i], x < pwlf_.fit_breaks[i + 1])))
        # Handle upper bound
        if upper_ == CONSTANT_KEY:
            eqn = RegressionProcessor.get_symbolic_eqn(pwlf_, pwlf_.n_segments, x)
            conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[-1]}), x >= pwlf_.fit_breaks[-1]))
        return conditions

    @staticmethod
    def get_symbolic_eqn(pwlf_: pwlf.PiecewiseLinFit, segment_number: int, x: Union[float, sp.Symbol]):
        """Get symbolic equation for a specific segment."""
        if pwlf_.degree < 1:
            raise ValueError('Degree must be at least 1')
        if segment_number < 1 or segment_number > pwlf_.n_segments:
            raise ValueError('segment_number not possible')
        my_eqn = 0
        # assemble degree = 1 first
        for line in range(segment_number):
            if line == 0:
                my_eqn = pwlf_.beta[0] + (pwlf_.beta[1]) * (x - pwlf_.fit_breaks[0])
            else:
                my_eqn += (pwlf_.beta[line + 1]) * (x - pwlf_.fit_breaks[line])
        # assemble all other degrees
        if pwlf_.degree > 1:
            for k in range(2, pwlf_.degree + 1):
                for line in range(segment_number):
                    beta_index = pwlf_.n_segments * (k - 1) + line + 1
                    my_eqn += (pwlf_.beta[beta_index]) * (x - pwlf_.fit_breaks[line]) ** k
        # Only call simplify if x is symbolic
        if isinstance(x, (sp.Symbol, sp.Expr)):
            return my_eqn.simplify()
        else:
            return my_eqn
