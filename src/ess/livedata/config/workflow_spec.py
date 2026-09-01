# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for data reduction workflow widget creation and configuration.

Two distinct parameter surfaces shape what a user sees, and conflating them is a
common source of confusion:

* **Workflow params** (this module, via :class:`WorkflowSpec.params`) change *what
  is computed* — TOA edges, coordinate mode, ranges, ROIs. Configured from the
  workflow card's gear in the Workflows tab.
* **Plot-display params** (``dashboard/plot_params.py``) change *how an
  already-computed output is displayed* — window, rate, scale, layout. Configured
  in the "Add layer" plot-creation wizard.

A given knob lives on exactly one surface, but nothing about a parameter's name
makes its surface obvious. The coupling that *is* declared lives on
:class:`OutputView.params`, which names the workflow params shaping each output and
drives the UI's param↔output cross-references in both surfaces.
"""

from __future__ import annotations

import enum
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeVar

import scipp as sc
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ess.livedata.core.timestamp import Timestamp

T = TypeVar('T')

JobNumber = uuid.UUID

Windowing = Literal['since_start', 'per_update']


class Temporality(enum.Enum):
    """How an output field's values relate to time.

    Declared per output field by annotating it :data:`WindowOutput`,
    :data:`CumulativeOutput` or :data:`SeriesOutput`, and read back with
    :meth:`WorkflowOutputsBase.temporality`::

        current: WindowOutput = pydantic.Field(...)

    Distinct from :data:`Windowing`, which selects *which* backing field a
    user-facing view exposes for the window mode the user picked. Temporality is
    a property of the field itself: what a single message means, and hence which
    operations over successive messages are valid. The two are not
    interchangeable — ``since_start`` is the default binding for fields that are
    not accumulations at all (ROI readbacks, static views).

    Every emitted DataArray carries a ``time`` coord: the instant its value
    refers to. What differs per temporality is what that instant means and what
    accompanies it, and this is the one place that says so. Code holding data
    but not the declaration sniffs for those coords; it is answering this
    question.
    """

    window = 'window'
    """Covers ``[start_time, time]``, where ``time`` is when the window closed.
    Successive windows are disjoint and advance. Summing or averaging over
    consecutive messages is meaningful, as is normalizing counts by the window
    duration."""

    cumulative = 'cumulative'
    """Value as observed at ``time``, accumulated since ``start_time``, which
    stays pinned for the lifetime of a job generation. Plotting the growth curve
    against ``time`` is meaningful; aggregating over successive messages
    double-counts."""

    series = 'series'
    """``time`` is the per-sample axis rather than a bound, and there is no
    ``start_time``. Messages are disjoint chunks of samples concatenated along
    that axis, so there is no single window duration to normalize by."""


WindowOutput = Annotated[sc.DataArray, Temporality.window]
"""An output field covering one update interval. See :attr:`Temporality.window`."""

CumulativeOutput = Annotated[sc.DataArray, Temporality.cumulative]
"""An output field accumulating over a job generation.

See :attr:`Temporality.cumulative`. This is also what a bare ``sc.DataArray``
field means, so reduction outputs need no annotation; spell it out where a
model mixes temporalities and the contrast carries information.
"""

SeriesOutput = Annotated[sc.DataArray, Temporality.series]
"""An output field carrying its own ``time`` axis. See :attr:`Temporality.series`."""


WINDOWING_FOR_TEMPORALITY: Mapping[Temporality, Windowing] = {
    Temporality.window: 'per_update',
    Temporality.series: 'per_update',
    Temporality.cumulative: 'since_start',
}
"""Which window mode each temporality backs.

``Windowing`` is a user-facing mode selector; ``Temporality`` is the property of
the field that makes it a valid answer for that mode. The mapping is total, so
views declare their member fields and the binding follows — there is no second
place to state it and therefore nothing to keep in sync.
"""


@dataclass(frozen=True)
class OutputView:
    """A user-facing output bundling one or more backend fields.

    Each view represents a single quantity (e.g. "Histogram", "Total") that may
    be observed over different time windows. ``fields`` names the backend
    pydantic fields carrying that quantity; which window mode each one serves
    follows from its declared :class:`Temporality` (see
    :data:`WINDOWING_FOR_TEMPORALITY`), so a view typically pairs a
    ``cumulative`` field with a ``window`` one. Resolution needs the outputs
    model and therefore lives on :class:`WorkflowSpec`, not here.

    ``params`` names the workflow parameter fields (top-level fields of the
    workflow's params model) that shape this output. It powers the UI's
    param↔output cross-references — "this output is controlled by these
    parameters" and the inverse. Names are resolved leniently against the
    actual params model: an output model may be shared across several params
    variants (e.g. a TOA-only variant lacking wavelength fields), so names
    absent from a given model are simply skipped.
    """

    name: str
    title: str
    fields: tuple[str, ...]
    description: str | None = None
    params: tuple[str, ...] = ()


class WorkflowOutputsBase(BaseModel):
    """Base class for all workflow output models.

    Provides common configuration for output models, including support for
    arbitrary types like scipp.DataArray. Subclasses may declare
    ``output_views`` as a ``ClassVar`` to bundle pydantic fields into
    user-facing views. When absent, a single view is derived per field.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_views: ClassVar[tuple[OutputView, ...]] = ()

    @classmethod
    def temporality(cls, field_name: str) -> Temporality:
        """Return the declared :class:`Temporality` of an output field.

        Fields that do not annotate one are ``cumulative``: a workflow output
        accumulates over the job's lifetime unless it explicitly resets each
        window, and the same default applies to outputs that are not
        accumulations at all (readbacks, static views) — nothing may be summed
        across their messages either.
        """
        for meta in cls.model_fields[field_name].metadata:
            if isinstance(meta, Temporality):
                return meta
        return Temporality.cumulative

    @classmethod
    def fields_with(cls, temporality: Temporality) -> tuple[str, ...]:
        """Return the names of output fields declaring the given temporality."""
        return tuple(
            name for name in cls.model_fields if cls.temporality(name) is temporality
        )


class DefaultOutputs(WorkflowOutputsBase):
    """Default outputs model for workflows that don't specify outputs.

    Provides a single 'result' field for simple workflows.
    """

    result: sc.DataArray = Field(title='Result', description='Workflow output.')


class WorkflowGroup(BaseModel, frozen=True):
    """Display-oriented grouping for workflow specs.

    Groups bundle related workflows for the dashboard UI. Identity is by
    the ``name`` field; the ``Literal`` constraint locks the set of legal
    categories to prevent ad-hoc construction sneaking in new ones.

    The ``name`` literals double as backend service names, and the coupling is
    load-bearing: :meth:`Instrument.register_spec` defaults a workflow's
    service to its group's ``name`` when no explicit ``service`` is given.
    """

    name: Literal['data_reduction', 'monitor_data', 'detector_data', 'timeseries']
    title: str
    description: str = ''


REDUCTION = WorkflowGroup(
    name='data_reduction',
    title='Reduction',
    description='Scientific data reduction workflows.',
)
MONITORS = WorkflowGroup(
    name='monitor_data',
    title='Monitors',
    description='Beam monitor data workflows.',
)
DETECTORS = WorkflowGroup(
    name='detector_data',
    title='Detectors',
    description='Detector data workflows.',
)
TIMESERIES = WorkflowGroup(
    name='timeseries',
    title='Devices and sensors',
    description=(
        'Time-stamped readings from motors, sample environment, and other '
        'instrument components. Stored as NXlog groups in NeXus files.'
    ),
)


class WorkflowId(BaseModel, frozen=True):
    instrument: str
    name: str
    version: int

    @model_validator(mode='before')
    @classmethod
    def _accept_string(cls, value: Any) -> Any:
        """Accept the ``instrument/name/version`` string form on input.

        This is the form exported in ``device_contract.yaml``; accepting it lets
        NICOS address a reset command by the workflow_id string it already holds,
        without reconstructing the nested object. The nested ``{instrument, name,
        version}`` form (used by the dashboard) passes through unchanged.
        """
        if isinstance(value, str):
            parts = value.split('/')
            if len(parts) != 3:
                raise ValueError(f"Invalid WorkflowId string format: {value}")
            instrument, name, version = parts
            return {'instrument': instrument, 'name': name, 'version': version}
        return value

    def __str__(self) -> str:
        return f"{self.instrument}/{self.name}/{self.version}"

    @staticmethod
    def from_string(workflow_id_str: str) -> WorkflowId:
        """Parse WorkflowId from string representation."""
        return WorkflowId.model_validate(workflow_id_str)


@dataclass(frozen=True, slots=True, kw_only=True)
class JobId:
    source_name: str
    job_number: JobNumber

    def __str__(self) -> str:
        """
        String representation for use in stream names and identifiers.

        Returns '{source_name}/{job_number}' to ensure unique identification
        across detectors in multi-detector workflows.
        """
        return f"{self.source_name}/{self.job_number}"


@dataclass(frozen=True)
class AuxInput:
    """Specification of an auxiliary data source input for a workflow.

    Each aux input defines a *role* that a workflow consumes (e.g.,
    "incident monitor for normalization") and maps it to one or more
    physical streams that can fill that role.

    The ``title`` and ``description`` describe the **role** — what the
    workflow uses this input for. Display names for the individual stream
    *choices* come from ``SourceMetadata`` on the ``Instrument``, which
    describes the physical component (e.g., position, hardware details).
    """

    choices: tuple[str, ...]
    default: str
    title: str = ''
    description: str = ''


class AuxSources:
    """Specification of auxiliary data source inputs for a workflow.

    Each input maps a logical name to one or more stream choices. A string
    shorthand creates a single fixed choice.

    Parameters
    ----------
    inputs:
        Mapping from logical input names to either a stream name string
        (single fixed choice) or an :class:`AuxInput` specification.
    """

    def __init__(self, inputs: dict[str, str | AuxInput]) -> None:
        self._inputs: dict[str, AuxInput] = {}
        for key, value in inputs.items():
            if isinstance(value, str):
                self._inputs[key] = AuxInput(choices=(value,), default=value)
            else:
                self._inputs[key] = value

    @property
    def inputs(self) -> dict[str, AuxInput]:
        """Return a copy of the inputs dict."""
        return dict(self._inputs)

    def get_defaults(self) -> dict[str, str]:
        """Return default selections for all inputs."""
        return {name: inp.default for name, inp in self._inputs.items()}

    def render(
        self,
        workflow_id: WorkflowId,
        source_name: str,
        selections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Render auxiliary source stream names for one workflow output.

        The default implementation returns the selected (or default) stream
        names unchanged. Subclasses can override to transform names, e.g. to
        scope a stream to the view it belongs to.

        Parameters
        ----------
        workflow_id:
            The workflow the job belongs to.
        source_name:
            The source the job processes. Together with ``workflow_id`` this
            is the view's stable identity; the job's ``job_number`` is
            deliberately not passed, so a name a subclass renders for itself
            stays addressable across restarts of the job.
        selections:
            User selections overriding defaults. Keys not present in the
            inputs specification are ignored.

        Returns
        -------
        :
            Mapping from input names to stream names for routing.
        """
        result = self.get_defaults()
        if selections:
            for key in result:
                if key in selections:
                    result[key] = selections[key]
        return result


class ResultKey(BaseModel, frozen=True):
    # Workflows produce one or more named outputs. Each output is serialized as a
    # separate da00 message. The output_name identifies which output this key refers to.
    workflow_id: WorkflowId = Field(description="Workflow ID")
    job_id: JobId = Field(description="Job ID")
    output_name: str = Field(
        default='result', description="Name of the workflow output"
    )

    @property
    def data_key(self) -> DataKey:
        """Stable dashboard key for this result, stripping the job_number."""
        return DataKey(
            workflow_id=self.workflow_id,
            source_name=self.job_id.source_name,
            output_name=self.output_name,
        )


class DataKey(BaseModel, frozen=True):
    """Stable identity of one workflow output stream.

    Unlike :py:class:`ResultKey` — the wire key, which embeds the per-commit
    ``job_number`` — this triple is stable across restarts of a workflow. It is
    the same identity used for stable extraction by NICOS device contracts
    (ADR 0006). The dashboard's data plane is keyed by ``DataKey``;
    ``job_number`` is consumed at the ingest filter as a generation token and
    recorded as a provenance stamp.
    """

    workflow_id: WorkflowId = Field(description="Workflow ID")
    source_name: str = Field(description="Name of the source")
    output_name: str = Field(
        default='result', description="Name of the workflow output"
    )


class NoParams(BaseModel):
    """
    Params model for workflows that take no configuration.

    Workflows always have a params model, so consumers never branch on its
    absence; "takes no parameters" is expressed as a model with no fields.
    Extra fields are rejected so that sending params to such a workflow is an
    error rather than silently ignored.
    """

    model_config = ConfigDict(extra='forbid')


class WorkflowSpec(BaseModel):
    """
    Model for workflow specification.

    This model is used to define a workflow and its parameters. The ESSlivedata
    dashboard uses these to create user interfaces for configuring workflows.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instrument: str = Field(
        description="Name of the instrument this workflow is associated with."
    )
    group: WorkflowGroup = Field(
        description=(
            "Display-oriented group this workflow belongs to. Carries the UI "
            "title and description; the ``name`` field is the canonical category "
            "identifier (one of ``data_reduction``, ``monitor_data``, "
            "``detector_data``, ``timeseries``)."
        ),
    )
    name: str = Field(description="Name of the workflow. Used internally.")
    version: int = Field(description="Version of the workflow.")
    title: str = Field(description="Title of the workflow. For display in the UI.")
    description: str = Field(description="Description of the workflow.")
    source_names: list[str] = Field(
        default_factory=list,
        description="List of detector/other streams the workflow can be applied to.",
    )
    aux_sources: AuxSources | None = Field(
        default=None,
        description=(
            "Auxiliary data source specification. Defines the available auxiliary "
            "data streams that a workflow can consume, with their choices, defaults, "
            "and UI metadata."
        ),
    )
    reset_on_run_transition: bool = Field(
        default=True,
        description=(
            "Whether jobs of this workflow should reset when an instrument run "
            "starts or stops. Set to False for workflows that accumulate across "
            "runs (e.g., timeseries)."
        ),
    )
    supports_reset: bool = Field(
        default=True,
        description=(
            "Whether jobs of this workflow support the manual reset action. "
            "When False, the dashboard disables the reset button and the "
            "backend rejects reset commands."
        ),
    )
    params: type[BaseModel] = Field(
        default=NoParams,
        description=(
            "Model for workflow params. Defaults to :class:`NoParams` for "
            "workflows that take no configuration."
        ),
    )
    outputs: type[WorkflowOutputsBase] = Field(
        default=DefaultOutputs,
        description=(
            "Pydantic model defining workflow outputs with their metadata. "
            "Field names are simplified identifiers (e.g., 'i_of_d_two_theta') "
            "that match keys returned by workflow.finalize(). Field types should "
            "be scipp.DataArray. Field metadata (title, description) provides "
            "human-readable names and explanations for the UI. "
            "\n\n"
            "IMPORTANT: Field definition order matters. The UI presents outputs in "
            "this order and auto-selects the first one, so put the primary output "
            "first. "
            "\n\n"
            "IMPORTANT: Use default_factory to provide an empty DataArray template "
            "with the correct structure (dims, coords, units). This enables the "
            "dashboard to perform automatic plotter selection before data exists. "
            "Example:\n"
            "    field: sc.DataArray = Field(\n"
            "        default_factory=lambda: sc.DataArray(\n"
            "            sc.zeros(dims=['x'], shape=[0], unit='counts'),\n"
            "            coords={'x': sc.arange('x', 0, unit='m')}\n"
            "        ),\n"
            "        title='Result',\n"
            "        description='Output description'\n"
            "    )"
        ),
    )
    device_outputs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Outputs exposed to NICOS as derived devices, mapping output field "
            "name to a device-name template. The template is formatted with "
            "``{source_name}`` once per entry in ``source_names``; include the "
            "placeholder whenever more than one source is declared, otherwise the "
            "rendered names collide. The per-instrument NICOS device list is "
            "generated from this declaration; see "
            ":mod:`ess.livedata.config.device_contract`."
        ),
    )

    @model_validator(mode='after')
    def validate_device_outputs(self) -> WorkflowSpec:
        """Validate that every declared device output is a real output field."""
        unknown = set(self.device_outputs) - set(self.outputs.model_fields)
        if unknown:
            raise ValueError(
                f"device_outputs references unknown output field(s) "
                f"{sorted(unknown)}; declared outputs: "
                f"{sorted(self.outputs.model_fields)}"
            )
        return self

    @field_validator('outputs', mode='after')
    @classmethod
    def validate_unique_output_titles(
        cls, outputs: type[WorkflowOutputsBase]
    ) -> type[WorkflowOutputsBase]:
        """Validate that user-facing view titles are unique within the workflow."""
        views = _resolve_output_views(outputs)
        title_counts: dict[str, list[str]] = defaultdict(list)
        for view in views:
            title_counts[view.title].append(view.name)

        duplicates = {
            title: names for title, names in title_counts.items() if len(names) > 1
        }
        if duplicates:
            dup_str = ", ".join(
                f"'{title}' (views: {', '.join(names)})"
                for title, names in duplicates.items()
            )
            raise ValueError(
                f"Output view titles must be unique within a workflow. "
                f"Duplicate titles found: {dup_str}"
            )

        return outputs

    @field_validator('outputs', mode='after')
    @classmethod
    def validate_unambiguous_windowing(
        cls, outputs: type[WorkflowOutputsBase]
    ) -> type[WorkflowOutputsBase]:
        """Validate that a view's fields back distinct window modes.

        Two fields of one view resolving to the same :data:`Windowing` would
        make the view ambiguous: selecting that mode could return either. The
        usual cause is a missing :class:`Temporality` annotation, since
        unannotated fields default to ``cumulative``.
        """
        for view in _resolve_output_views(outputs):
            by_windowing: dict[Windowing, list[str]] = defaultdict(list)
            for field_name in view.fields:
                windowing = WINDOWING_FOR_TEMPORALITY[outputs.temporality(field_name)]
                by_windowing[windowing].append(field_name)
            for windowing, field_names in by_windowing.items():
                if len(field_names) > 1:
                    raise ValueError(
                        f"Output view '{view.name}' has multiple fields backing "
                        f"'{windowing}': {sorted(field_names)}. Annotate their "
                        f"Temporality so each field backs a distinct window mode."
                    )

        return outputs

    def get_id(self) -> WorkflowId:
        """
        Get a unique identifier for the workflow.

        The identifier is a combination of instrument, name, and version.
        """
        return WorkflowId(
            instrument=self.instrument,
            name=self.name,
            version=self.version,
        )

    def get_output_views(self) -> Sequence[OutputView]:
        """Return the user-facing output views for this workflow.

        Falls back to one view per pydantic field when the outputs class does
        not declare ``output_views``.
        """
        return _resolve_output_views(self.outputs)

    def get_output_view(self, view_name: str) -> OutputView | None:
        """Return the named output view, or None if not found."""
        for view in self.get_output_views():
            if view.name == view_name:
                return view
        return None

    def _windowing_by_field(self, view_name: str) -> dict[Windowing, str]:
        """Map each window mode the named view backs to the field serving it.

        Derived from each member field's :class:`Temporality`; empty for an
        unknown view. Distinctness is enforced at registration by
        :meth:`validate_unambiguous_windowing`.
        """
        view = self.get_output_view(view_name)
        if view is None:
            return {}
        return {
            WINDOWING_FOR_TEMPORALITY[self.outputs.temporality(field_name)]: field_name
            for field_name in view.fields
        }

    def windowing_options(self, view_name: str) -> frozenset[Windowing]:
        """Return the window modes the named view has a real backing field for."""
        return frozenset(self._windowing_by_field(view_name))

    def field_for(self, view_name: str, windowing: Windowing) -> str:
        """Return the backend field a view exposes for the requested windowing.

        Falls back to the view's other field when the requested windowing has
        no backing field — handles views exposing only one flavor (e.g.
        cumulative-only quantities). Unknown views resolve to their own name,
        which is the field name for workflows that declare no ``output_views``.
        """
        by_windowing = self._windowing_by_field(view_name)
        if not by_windowing:
            return view_name
        if (field_name := by_windowing.get(windowing)) is not None:
            return field_name
        return next(iter(by_windowing.values()))

    def get_output_template(self, view_name: str) -> sc.DataArray | None:
        """Get a template DataArray for the specified output view.

        Returns the ``default_factory`` template of the view's canonical
        backing field (the ``since_start`` one if present, else its other).
        Templates are empty DataArrays demonstrating the expected structure
        (dims, coords, units), used by the dashboard for plotter selection
        before any data has arrived.

        Returns None if the view is unknown or its canonical field has no
        ``default_factory``.
        """
        if self.get_output_view(view_name) is None:
            return None
        field_info = self.outputs.model_fields.get(
            self.field_for(view_name, 'since_start')
        )
        if field_info is None or not field_info.default_factory:
            return None
        return field_info.default_factory()

    def get_output_param_titles(self, view_name: str) -> list[str]:
        """Titles of the param groups that shape the given output view.

        Resolves the view's ``params`` (param field names) against the
        workflow's params model, skipping names absent from this particular
        model (output views are shared across params-model variants). Titles
        are returned in params-model field order for stable display.
        """
        view = self.get_output_view(view_name)
        if view is None or self.params is None or not view.params:
            return []
        wanted = set(view.params)
        return [
            (field_info.title or name)
            for name, field_info in self.params.model_fields.items()
            if name in wanted
        ]

    def get_param_output_titles(self) -> dict[str, list[str]]:
        """Map each param field name to the titles of the outputs it shapes.

        Inverse of :attr:`OutputView.params`, used to annotate parameter
        groups in the configuration UI with the outputs they affect. Param
        names not present in the params model are omitted.
        """
        param_fields = set(self.params.model_fields) if self.params else set()
        result: dict[str, list[str]] = {}
        for view in self.get_output_views():
            for name in view.params:
                if name not in param_fields:
                    continue
                titles = result.setdefault(name, [])
                if view.title not in titles:
                    titles.append(view.title)
        return result


@dataclass
class JobSchedule:
    """
    Defines when a job should start and optionally when it should end.

    All timestamps are in nanoseconds since the epoch (UTC) and reference the timestamps
    of the raw data being processed (as opposed to when it should be processed).
    """

    start_time: Timestamp | None = None  # When job should start processing
    end_time: Timestamp | None = None  # When job should stop (None = no limit)

    def __post_init__(self) -> None:
        """Validate the schedule configuration."""
        if (
            self.end_time is not None
            and self.start_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError(
                f"Job end_time={self.end_time} must be greater than start_time="
                f"{self.start_time}, or start_time must be None (immediate start)"
            )

    def should_start(self, current_time: Timestamp) -> bool:
        """
        Check if the job should start based on the current time.

        Returns True if the job should start, False otherwise.
        """
        return self.start_time is None or current_time >= self.start_time


class WorkflowConfig(BaseModel):
    """
    Model for workflow configuration.

    This model is used to set the parameter values for a specific workflow. The values
    correspond to the parameters defined in the workflow specification
    :py:class:`WorkflowSpec`.

    Note on message_id vs job_number
    --------------------------------
    These two identifiers serve fundamentally different purposes:

    - ``message_id``: Transient identifier for command/response correlation
      (ACK pattern). Frontend generates it, backend echoes it in
      CommandAcknowledgement. Discarded once ACK is received. Used only for
      the request/response handshake.

    - ``job_number``: Persistent job identity for the entire job lifecycle.
      Used in JobId for result routing, job commands (stop/reset), and data
      correlation.

    This message conflates "configure" and "start" into a single command, so both
    fields are present. Splitting them was considered and dropped: it would let several
    clients start independent jobs from one shared configuration, and there is only one
    client that starts jobs. NICOS consumes contracted outputs and resets them, keyed by
    ``WorkflowId`` (ADR 0006); it does not configure or start.
    """

    kind: Literal['workflow_config'] = 'workflow_config'
    identifier: WorkflowId = Field(
        description="Hash of the workflow, used to identify the workflow."
    )
    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Transient identifier for command/response correlation. An initiator that "
            "tracks the reply sets this; one that does not may omit it, in which case "
            "an id is generated for the ack it ignores. Distinct from job_id which "
            "identifies the job itself."
        ),
    )
    job_id: JobId = Field(
        description=(
            "Identity of the job this config starts. Carries both source_name and "
            "job_number; used for result routing and job control commands. Distinct "
            "from message_id which is only for command acknowledgement correlation."
        ),
    )
    schedule: JobSchedule = Field(
        default_factory=JobSchedule, description="Schedule for the workflow."
    )
    aux_source_names: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Selected auxiliary source names as a mapping from field name (as defined "
            "in WorkflowSpec.aux_sources) to the selected stream name."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the workflow, as JSON-serialized Pydantic model.",
    )

    @classmethod
    def from_params(
        cls,
        workflow_id: WorkflowId,
        job_id: JobId,
        message_id: str | None = None,
        params: dict | None = None,
        aux_source_names: dict | None = None,
    ) -> WorkflowConfig:
        """
        Create a WorkflowConfig from parameters.

        Parameters
        ----------
        workflow_id:
            Identifier for the workflow
        job_id:
            Identity of the job this config starts.
        message_id:
            Message ID for command acknowledgement correlation. Omit to let an id
            be generated for an ack the initiator does not consume.
        params:
            Workflow parameters as dict, or None if no params
        aux_source_names:
            Auxiliary source selections as dict, or None if no aux sources

        Returns
        -------
        :
            WorkflowConfig instance ready to be sent to backend
        """
        fields: dict[str, Any] = {
            "identifier": workflow_id,
            "job_id": job_id,
            "aux_source_names": aux_source_names or {},
            "params": params or {},
        }
        if message_id is not None:
            fields["message_id"] = message_id
        return cls(**fields)


_CORRELATABLE = (Temporality.window, Temporality.series)


def _is_correlatable(outputs: type[WorkflowOutputsBase], field_name: str) -> bool:
    """Check whether an output field can serve as a correlation axis.

    Requires a scalar quantity that advances along wall-clock time: a per-window
    value or a timestamped series. Cumulative fields are excluded — correlating
    against a monotonically growing total says more about the run length than
    about the quantity.
    """
    field_info = outputs.model_fields.get(field_name)
    if field_info is None or not field_info.default_factory:
        return False
    if field_info.default_factory().ndim != 0:
        return False
    return outputs.temporality(field_name) in _CORRELATABLE


def _resolve_output_views(
    outputs: type[WorkflowOutputsBase],
) -> tuple[OutputView, ...]:
    """Return the declared ``output_views`` or a default one-view-per-field set.

    When an outputs class does not declare ``output_views``, each pydantic
    field becomes its own view, with the bare field name used as both the view
    name and the backing field. This keeps reduction-style Outputs classes
    working without annotation.
    """
    declared = getattr(outputs, 'output_views', ())
    if declared:
        return tuple(declared)
    return tuple(
        OutputView(
            name=field_name,
            title=(field_info.title or field_name),
            fields=(field_name,),
            description=field_info.description,
        )
        for field_name, field_info in outputs.model_fields.items()
    )


def find_timeseries_outputs(
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
) -> list[tuple[WorkflowId, str, str]]:
    """
    Find all timeseries output views in the workflow registry.

    A timeseries output is a 0-D field declaring :attr:`Temporality.window` or
    :attr:`Temporality.series` — a scalar that advances along wall-clock time.
    Every backing field of an output view is inspected; views with at least one
    matching field are reported by view name.

    Parameters
    ----------
    workflow_registry:
        Registry of workflow specs to search.

    Returns
    -------
    :
        List of (workflow_id, source_name, view_name) tuples for all
        timeseries views found. Each source_name from the workflow's
        source_names list is paired with each timeseries view name.
    """
    results: list[tuple[WorkflowId, str, str]] = []

    for workflow_id, spec in workflow_registry.items():
        if spec.outputs is None:
            continue

        timeseries_views: list[str] = []
        for view in spec.get_output_views():
            for field_name in view.fields:
                if _is_correlatable(spec.outputs, field_name):
                    timeseries_views.append(view.name)
                    break

        results.extend(
            (workflow_id, source_name, view_name)
            for source_name in spec.source_names
            for view_name in timeseries_views
        )

    return results
