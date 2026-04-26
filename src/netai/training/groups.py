"""Private group/network training with resource gating and access control."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from netai.crypto.identity import GroupKey, NodeIdentity, derive_group_key
from netai.training.voting import (
    VotingEngine, ResourcePledge, VoteType, ProposalStatus, VoteWeight,
)

logger = logging.getLogger(__name__)


class GroupVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SECRET = "secret"


class MemberRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    OBSERVER = "observer"


class GroupPolicy(BaseModel):
    min_resource_score: float = 0.0
    min_reputation: float = 0.0
    require_approval: bool = True
    max_members: int = 100
    vote_threshold_start: float = 0.5
    vote_threshold_stop: float = 0.6
    vote_quorum: float = 0.3
    voting_period_hours: float = 72.0
    auto_accept_pledges: bool = False
    allowed_architectures: list[str] = Field(default_factory=list)
    max_concurrent_jobs: int = 5
    checkpoint_visibility: str = "members"


class Member(BaseModel):
    user_id: str
    node_id: str = ""
    role: MemberRole = MemberRole.MEMBER
    joined_at: float = Field(default_factory=time.time)
    pledge: ResourcePledge | None = None
    reputation: float = 1.0
    invited_by: str = ""
    status: str = "active"


class TrainingGroup(BaseModel):
    group_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    visibility: GroupVisibility = GroupVisibility.PRIVATE
    owner_id: str = ""
    members: dict[str, Member] = Field(default_factory=dict)
    policy: GroupPolicy = Field(default_factory=GroupPolicy)
    created_at: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)
    active_jobs: list[str] = Field(default_factory=list)
    total_compute_hours: float = 0.0
    models_trained: int = 0
    encrypted_key: str = ""


class GroupManager:
    def __init__(self):
        self.groups: dict[str, TrainingGroup] = {}
        self.user_groups: dict[str, list[str]] = {}
        self.group_keys: dict[str, GroupKey] = {}
        self.node_identities: dict[str, NodeIdentity] = {}
        self.voting: VotingEngine = VotingEngine()
        self._invite_codes: dict[str, dict] = {}

    def create_group(
        self,
        name: str,
        owner_id: str,
        description: str = "",
        visibility: GroupVisibility = GroupVisibility.PRIVATE,
        policy: GroupPolicy | None = None,
        tags: list[str] | None = None,
        passphrase: str | None = None,
    ) -> TrainingGroup:
        group = TrainingGroup(
            name=name,
            owner_id=owner_id,
            description=description,
            visibility=visibility,
            policy=policy or GroupPolicy(),
            tags=tags or [],
        )
        group.members[owner_id] = Member(
            user_id=owner_id,
            role=MemberRole.OWNER,
            joined_at=time.time(),
        )
        group_key = derive_group_key(group.group_id, passphrase)
        self.group_keys[group.group_id] = group_key
        self.groups[group.group_id] = group
        self.user_groups.setdefault(owner_id, []).append(group.group_id)
        logger.info("Group created: %s (%s) by %s", name, group.group_id, owner_id)
        return group

    def join_group(
        self,
        group_id: str,
        user_id: str,
        node_id: str = "",
        pledge: ResourcePledge | None = None,
        invite_code: str | None = None,
    ) -> tuple[bool, str]:
        group = self.groups.get(group_id)
        if not group:
            return False, "Group not found"
        if user_id in group.members:
            return False, "Already a member"
        if len(group.members) >= group.policy.max_members:
            return False, "Group is full"
        if pledge and group.policy.min_resource_score > 0:
            if pledge.compute_score < group.policy.min_resource_score:
                return False, f"Resource pledge below minimum ({group.policy.min_resource_score:.1f} required)"
        if group.policy.require_approval:
            if invite_code:
                inv = self._invite_codes.get(invite_code)
                if not inv or inv["group_id"] != group_id:
                    return False, "Invalid invite code"
                if time.time() > inv.get("expires", 0):
                    return False, "Invite code expired"
                inv["uses"] = inv.get("uses", 0) + 1
                if inv.get("max_uses", 1) > 0 and inv["uses"] >= inv["max_uses"]:
                    del self._invite_codes[invite_code]
            elif group.visibility != GroupVisibility.PUBLIC:
                return False, "Invite code required for private groups"
        member = Member(
            user_id=user_id,
            node_id=node_id,
            role=MemberRole.MEMBER,
            pledge=pledge,
        )
        group.members[user_id] = member
        self.user_groups.setdefault(user_id, []).append(group.group_id)
        if pledge:
            self.voting.pledges[user_id] = pledge
        logger.info("User %s joined group %s", user_id, group.name)
        return True, "Joined successfully"

    def leave_group(self, group_id: str, user_id: str) -> bool:
        group = self.groups.get(group_id)
        if not group or user_id not in group.members:
            return False
        if user_id == group.owner_id:
            return False
        del group.members[user_id]
        if user_id in self.user_groups:
            self.user_groups[user_id] = [g for g in self.user_groups[user_id] if g != group_id]
        return True

    def create_invite(self, group_id: str, inviter_id: str, max_uses: int = 1, expires_hours: float = 72.0) -> str | None:
        group = self.groups.get(group_id)
        if not group:
            return None
        inviter = group.members.get(inviter_id)
        if not inviter or inviter.role not in (MemberRole.OWNER, MemberRole.ADMIN):
            return None
        code = secrets.token_urlsafe(16)
        self._invite_codes[code] = {
            "group_id": group_id,
            "inviter_id": inviter_id,
            "max_uses": max_uses,
            "uses": 0,
            "expires": time.time() + expires_hours * 3600,
        }
        return code

    def remove_member(self, group_id: str, remover_id: str, target_id: str) -> tuple[bool, str]:
        group = self.groups.get(group_id)
        if not group:
            return False, "Group not found"
        remover = group.members.get(remover_id)
        if not remover or remover.role not in (MemberRole.OWNER, MemberRole.ADMIN):
            return False, "Insufficient permissions"
        target = group.members.get(target_id)
        if not target:
            return False, "Member not found"
        if target_id == group.owner_id:
            return False, "Cannot remove owner"
        if remover.role == MemberRole.ADMIN and target.role in (MemberRole.OWNER, MemberRole.ADMIN):
            return False, "Cannot remove equal or higher role"
        group.members.pop(target_id)
        return True, "Member removed"

    def set_member_role(self, group_id: str, setter_id: str, target_id: str, role: MemberRole) -> tuple[bool, str]:
        group = self.groups.get(group_id)
        if not group:
            return False, "Group not found"
        setter = group.members.get(setter_id)
        if not setter or setter.role != MemberRole.OWNER:
            return False, "Only owner can change roles"
        target = group.members.get(target_id)
        if not target:
            return False, "Member not found"
        group.members[target_id].role = role
        return True, f"Role set to {role.value}"

    def propose_training(
        self,
        group_id: str,
        proposer_id: str,
        job_config: dict[str, Any],
        voting_deadline: float | None = None,
    ) -> tuple[Any, str]:
        group = self.groups.get(group_id)
        if not group:
            return None, "Group not found"
        if proposer_id not in group.members:
            return None, "Not a member"
        if group.policy.allowed_architectures:
            arch = job_config.get("model_architecture", "")
            if arch and arch not in group.policy.allowed_architectures:
                return None, f"Architecture {arch} not allowed"
        if len(group.active_jobs) >= group.policy.max_concurrent_jobs:
            return None, "Max concurrent jobs reached"
        deadline = voting_deadline or (time.time() + group.policy.voting_period_hours * 3600)
        proposal = self.voting.create_train_proposal(
            job_config=job_config,
            proposer_id=proposer_id,
            group_id=group_id,
            voting_deadline=deadline,
        )
        proposal.quorum = group.policy.vote_quorum
        proposal.threshold = group.policy.vote_threshold_start
        proposal.vote_weight_type = VoteWeight.BY_RESOURCE
        return proposal, proposal.proposal_id

    def approve_training(self, group_id: str, proposal_id: str) -> dict[str, Any]:
        group = self.groups.get(group_id)
        if not group:
            return {"error": "Group not found"}
        proposal = self.voting.get_proposal(proposal_id)
        if not proposal:
            return {"error": "Proposal not found"}
        if proposal.group_id != group_id:
            return {"error": "Proposal not in this group"}
        result = proposal.result
        if result == ProposalStatus.PASSED:
            job_id = proposal.config.get("job_id", uuid.uuid4().hex[:12])
            group.active_jobs.append(job_id)
            return {"status": "approved", "job_id": job_id, "config": proposal.config}
        elif result == ProposalStatus.FAILED:
            return {"status": "rejected", "reason": "Vote failed or quorum not met"}
        return {"status": "pending", "votes_for": proposal.weighted_for, "votes_against": proposal.weighted_against}

    def validate_resource_access(self, group_id: str, user_id: str, required_gpu: int = 0, required_cpu: int = 0, required_ram_gb: float = 0) -> tuple[bool, str]:
        group = self.groups.get(group_id)
        if not group:
            return False, "Group not found"
        member = group.members.get(user_id)
        if not member:
            return False, "Not a member"
        if member.role == MemberRole.OBSERVER:
            return False, "Observers cannot start training"
        available_cpu = 0
        available_gpu = 0
        available_ram = 0.0
        for m in group.members.values():
            if m.pledge and m.status == "active":
                available_cpu += m.pledge.cpu_cores
                available_gpu += m.pledge.gpu_count
                available_ram += m.pledge.ram_gb
        if required_gpu > 0 and available_gpu < required_gpu:
            return False, f"Insufficient GPU resources ({available_gpu}/{required_gpu})"
        if required_cpu > 0 and available_cpu < required_cpu:
            return False, f"Insufficient CPU resources ({available_cpu}/{required_cpu})"
        if required_ram_gb > 0 and available_ram < required_ram_gb:
            return False, f"Insufficient RAM ({available_ram:.1f}/{required_ram_gb:.1f} GB)"
        return True, "Resources available"

    def get_group_resources(self, group_id: str) -> dict[str, Any]:
        group = self.groups.get(group_id)
        if not group:
            return {}
        members = [m for m in group.members.values() if m.status == "active"]
        pledged_cpu = sum(m.pledge.cpu_cores for m in members if m.pledge)
        pledged_gpu = sum(m.pledge.gpu_count for m in members if m.pledge)
        pledged_ram = sum(m.pledge.ram_gb for m in members if m.pledge)
        pledged_vram = sum(v for m in members if m.pledge for v in (m.pledge.gpu_vram_mb or []))
        return {
            "group_id": group_id,
            "group_name": group.name,
            "members": len(members),
            "pledged_cpu_cores": pledged_cpu,
            "pledged_gpu_count": pledged_gpu,
            "pledged_ram_gb": pledged_ram,
            "pledged_vram_mb": pledged_vram,
            "active_jobs": len(group.active_jobs),
            "max_concurrent_jobs": group.policy.max_concurrent_jobs,
            "total_compute_hours": group.total_compute_hours,
            "models_trained": group.models_trained,
            "member_details": [
                {
                    "user_id": m.user_id,
                    "role": m.role.value,
                    "pledge": m.pledge.summary if m.pledge else "No pledge",
                    "reputation": m.reputation,
                }
                for m in members
            ],
        }

    def list_groups(self, user_id: str | None = None, visibility: GroupVisibility | None = None) -> list[dict]:
        results = []
        for g in self.groups.values():
            if visibility and g.visibility != visibility:
                continue
            if g.visibility == GroupVisibility.SECRET:
                if user_id and user_id not in g.members:
                    continue
            results.append({
                "group_id": g.group_id,
                "name": g.name,
                "description": g.description,
                "visibility": g.visibility.value,
                "members": len(g.members),
                "tags": g.tags,
                "created_at": g.created_at,
            })
        return results

    def get_group(self, group_id: str) -> TrainingGroup | None:
        return self.groups.get(group_id)

    def can_start_training(self, group_id: str, user_id: str) -> tuple[bool, str]:
        group = self.groups.get(group_id)
        if not group:
            return False, "Group not found"
        member = group.members.get(user_id)
        if not member:
            return False, "Not a member"
        if member.role == MemberRole.OBSERVER:
            return False, "Observers cannot start training"
        if len(group.active_jobs) >= group.policy.max_concurrent_jobs:
            return False, "Max concurrent jobs reached"
        active_pledges = sum(1 for m in group.members.values() if m.pledge and m.status == "active")
        if active_pledges < 1:
            return False, "No active resource pledges"
        return True, "Can start training"