
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any

import numpy as np
import symengine as sym

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
from qiskit.pulse.library import *
def _lifted_gaussian(
    t: sym.Symbol,
    center: sym.Symbol | sym.Expr | complex,
    t_zero: sym.Symbol | sym.Expr | complex,
    sigma: sym.Symbol | sym.Expr | complex,
) -> sym.Expr:
    r"""Helper function that returns a lifted Gaussian symbolic equation.

    For :math:`\sigma=` ``sigma`` the symbolic equation will be

    .. math::

        f(x) = \exp\left(-\frac12 \left(\frac{x - \mu}{\sigma}\right)^2 \right),

    with the center :math:`\mu=` ``duration/2``.
    Then, each output sample :math:`y` is modified according to:

    .. math::

        y \mapsto \frac{y-y^*}{1.0-y^*},

    where :math:`y^*` is the value of the un-normalized Gaussian at the endpoints of the pulse.
    This sets the endpoints to :math:`0` while preserving the amplitude at the center,
    i.e. :math:`y` is set to :math:`1.0`.

    Args:
        t: Symbol object representing time.
        center: Symbol or expression representing the middle point of the samples.
        t_zero: The value of t at which the pulse is lowered to 0.
        sigma: Symbol or expression representing Gaussian sigma.

    Returns:
        Symbolic equation.
    """
    # Sympy automatically does expand.
    # This causes expression inconsistency after qpy round-trip serializing through sympy.
    # See issue for details: https://github.com/symengine/symengine.py/issues/409
    t_shifted = (t - center).expand()
    t_offset = (t_zero - center).expand()

    gauss = sym.exp(-((t_shifted / sigma) ** 2) / 2)
    offset = sym.exp(-((t_offset / sigma) ** 2) / 2)

    return (gauss - offset) / (1 - offset)


@functools.lru_cache(maxsize=None)
def _is_amplitude_valid(
    envelope_lam: Callable, time: tuple[float, ...], *fargs: float
) -> bool | np.bool_:
    """A helper function to validate maximum amplitude limit.

    Result is cached for better performance.

    Args:
        envelope_lam: The SymbolicPulse's lambdified envelope_lam expression.
        time: The SymbolicPulse's time array, given as a tuple for hashability.
        fargs: The arguments for the lambdified envelope_lam, as given by `_get_expression_args`,
            except for the time array.

    Returns:
        Return True if no sample point exceeds 1.0 in absolute value.
    """

    time = np.asarray(time, dtype=float)
    samples_norm = np.abs(envelope_lam(time, *fargs))
    epsilon = 1e-7  # The value of epsilon mimics that of Waveform._clip()
    return np.all(samples_norm < 1.0 + epsilon)


def _get_expression_args(expr: sym.Expr, params: dict[str, float]) -> list[np.ndarray | float]:
    """A helper function to get argument to evaluate expression.

    Args:
        expr: Symbolic expression to evaluate.
        params: Dictionary of parameter, which is a superset of expression arguments.

    Returns:
        Arguments passed to the lambdified expression.

    Raises:
        PulseError: When a free symbol value is not defined in the pulse instance parameters.
    """
    args: list[np.ndarray | float] = []
    for symbol in sorted(expr.free_symbols, key=lambda s: s.name):
        if symbol.name == "t":
            # 't' is a special parameter to represent time vector.
            # This should be place at first to broadcast other parameters
            # in symengine lambdify function.
            times = np.arange(0, params["duration"]) + 1 / 2
            args.insert(0, times)
            continue
        try:
            args.append(params[symbol.name])
        except KeyError as ex:
            raise PulseError(
                f"Pulse parameter '{symbol.name}' is not defined for this instance. "
                "Please check your waveform expression is correct."
            ) from ex
    return args


class _PulseType(type):
    """Metaclass to warn at isinstance check."""

    def __instancecheck__(cls, instance):
        cls_alias = getattr(cls, "alias", None)

        # TODO promote this to Deprecation warning in future.
        #  Once type information usage is removed from user code,
        #  we will convert pulse classes into functions.
        warnings.warn(
            "Typechecking with the symbolic pulse subclass will be deprecated. "
            f"'{cls_alias}' subclass instance is turned into SymbolicPulse instance. "
            f"Use self.pulse_type == '{cls_alias}' instead.",
            PendingDeprecationWarning,
        )

        if not isinstance(instance, SymbolicPulse):
            return False
        return instance.pulse_type == cls_alias

    def __getattr__(cls, item):
        # For pylint. A SymbolicPulse subclass must implement several methods
        # such as .get_waveform and .validate_parameters.
        # In addition, they conventionally offer attribute-like access to the pulse parameters,
        # for example, instance.amp returns instance._params["amp"].
        # If pulse classes are directly instantiated, pylint yells no-member
        # since the pulse class itself implements nothing. These classes just
        # behave like a factory by internally instantiating the SymbolicPulse and return it.
        # It is not realistic to write disable=no-member across qiskit packages.
        return NotImplemented



class custom_GaussianSquare(metaclass=_PulseType):
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Exactly one of the ``risefall_sigma_ratio`` and ``width`` parameters has to be specified.

    If ``risefall_sigma_ratio`` is not None and ``width`` is None:

    .. math::

        \\begin{aligned}
        \\text{risefall} &= \\text{risefall\\_sigma\\_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}
        \\end{aligned}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    In both cases, the lifted gaussian square pulse :math:`f'(x)` is defined as:

    .. math::

        \\begin{aligned}
        f'(x) &= \\begin{cases}\
            \\exp\\biggl(-\\frac12 \\frac{(x - \\text{risefall})^2}{\\text{sigma}^2}\\biggr)\
                & x < \\text{risefall}\\\\
            1\
                & \\text{risefall} \\le x < \\text{risefall} + \\text{width}\\\\
            \\exp\\biggl(-\\frac12\
                    \\frac{{\\bigl(x - (\\text{risefall} + \\text{width})\\bigr)}^2}\
                          {\\text{sigma}^2}\
                    \\biggr)\
                & \\text{risefall} + \\text{width} \\le x\
        \\end{cases}\\\\
        f(x) &= \\text{A} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)},\
            \\quad 0 \\le x < \\text{duration}
        \\end{aligned}

    where :math:`f'(x)` is the gaussian square waveform without lifting or amplitude scaling, and
    :math:`\\text{A} = \\text{amp} \\times \\exp\\left(i\\times\\text{angle}\\right)`.
    """

    alias = "GaussianSquare_custom"

    def __new__(
        cls,
        duration: int | ParameterValueType,
        amp: ParameterValueType,
        sigma: ParameterValueType,
        width: ParameterValueType | None = None,
        angle: ParameterValueType = 0.0,
        risefall_sigma_ratio: ParameterValueType | None = None,
        offset : ParameterValueType = 0.0,
        name: str | None = None,
        limit_amplitude: bool | None = None,
    ) -> ScalableSymbolicPulse:
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The magnitude of the amplitude of the Gaussian and square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class
                   docstring for more details.
            width: The duration of the embedded square pulse.
            angle: The angle of the complex amplitude of the pulse. Default value 0.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Returns:
            ScalableSymbolicPulse instance.

        Raises:
            PulseError: When width and risefall_sigma_ratio are both empty or both non-empty.
        """
        # Convert risefall_sigma_ratio into width which is defined in OpenPulse spec
        if width is None and risefall_sigma_ratio is None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter must be specified."
            )
        if width is not None and risefall_sigma_ratio is not None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter can be specified"
                " but not both."
            )
        if width is None and risefall_sigma_ratio is not None:
            width = duration - 2.0 * risefall_sigma_ratio * sigma

        parameters = {"sigma": sigma, "width": width}

        # Prepare symbolic expressions
        _t, _duration, _amp, _sigma, _width, _angle = sym.symbols(
            "t, duration, amp, sigma, width, angle"
        )
        _center = _duration / 2

        _sq_t0 = _center - _width / 2
        _sq_t1 = _center + _width / 2

        _gaussian_ledge = _lifted_gaussian(_t, _sq_t0, -1, _sigma)
        _gaussian_redge = _lifted_gaussian(_t, _sq_t1, _duration + 1, _sigma)

        envelope_expr = (
            _amp
            * sym.exp(sym.I * _angle)
            * sym.Piecewise(
                (_gaussian_ledge, _t <= _sq_t0), (_gaussian_redge, _t >= _sq_t1), (1, True)
            )
            * sym.exp(sym.I *_t*offset*2*np.pi)
        )

        consts_expr = sym.And(_sigma > 0, _width >= 0, _duration >= _width)
        valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0

        return ScalableSymbolicPulse(
            pulse_type=cls.alias,
            duration=duration,
            amp=amp,
            angle=angle,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
            envelope=envelope_expr,
            constraints=consts_expr,
            valid_amp_conditions=valid_amp_conditions_expr,
        )