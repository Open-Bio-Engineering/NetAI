"""Consensus protocol for distributed pipeline stage assignment."""

from netai.consensus.pipeline_coordinator import (
    PipelineCoordinator,
    PipelineProposal,
)

__all__ = ["PipelineCoordinator", "PipelineProposal"]
