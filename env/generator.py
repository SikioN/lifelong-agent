"""Synthetic support-ticket environment for DeMem-lite.

Each ticket has a descriptive topic (surface text, paraphrased) and a
ground-truth correct_action. For tenants in the "override" regime, the
correct action is remapped through a fixed derangement permutation to a
*different* topic's action — this decouples semantic similarity (driven by
topic content) from decision utility (driven by correct_action), per the
alpha-permutation Decoupled Bandit construction in docs/materials/PLAN.md.
"""
import numpy as np

N_TOPICS = 8
ACTION_LABELS = [f"ACTION_{i}" for i in range(N_TOPICS)]


def build_default_action_map(seed: int) -> dict[int, str]:
    """Random (seeded) bijection topic_id -> action label.

    Deliberately non-obvious: action labels are abstract codes, not
    semantically related to topic content, so a pretrained LM cannot guess
    the rule from world knowledge alone (see PLAN.md "утечка знаний из
    претрейна" risk).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_TOPICS)
    return {topic_id: ACTION_LABELS[perm[topic_id]] for topic_id in range(N_TOPICS)}


def _random_derangement(n: int, rng: np.random.Generator) -> list[int]:
    """A permutation of range(n) with no fixed points. Rejection sampling —
    converges in a handful of draws on average (P(derangement) -> 1/e)."""
    while True:
        perm = rng.permutation(n)
        if all(perm[i] != i for i in range(n)):
            return perm.tolist()


def build_mismatch_permutation(seed: int) -> dict[int, int]:
    """Fixed derangement topic_id -> topic_id used to compute the override
    regime's effective decision-identity. No fixed points, so override
    *always* routes to a genuinely different topic's action."""
    rng = np.random.default_rng(seed)
    derangement = _random_derangement(N_TOPICS, rng)
    return {topic_id: derangement[topic_id] for topic_id in range(N_TOPICS)}


TOPIC_TEMPLATES: dict[int, list[str]] = {
    0: [  # billing_dispute
        "billing invoice charge dispute: duplicate",
        "billing invoice charge dispute: incorrect",
        "billing invoice charge dispute: unrecognized",
        "billing invoice charge dispute: overcharge",
    ],
    1: [  # password_reset
        "password login reset: link expired",
        "password login reset: email missing",
        "password login reset: authentication failed",
        "password login reset: login locked",
    ],
    2: [  # data_export_request
        "data export download: full history",
        "data export download: usage records",
        "data export download: compliance archive",
        "data export download: csv file",
    ],
    3: [  # account_deletion
        "account deletion permanent: close service",
        "account deletion permanent: erase logs",
        "account deletion permanent: cancel subscription",
        "account deletion permanent: remove profile",
    ],
    4: [  # refund_request
        "refund money purchase: defective item",
        "refund money purchase: broken goods",
        "refund money purchase: faulty product",
        "refund money purchase: unsatisfied result",
    ],
    5: [  # shipping_delay
        "shipping delivery delay: tracking stuck",
        "shipping delivery delay: package late",
        "shipping delivery delay: order overdue",
        "shipping delivery delay: shipment lost",
    ],
    6: [  # feature_request
        "feature request add: dark mode toggle",
        "feature request add: notification toggle",
        "feature request add: shortcut toggle",
        "feature request add: widget toggle",
    ],
    7: [  # bug_report
        "bug report app crash: startup",
        "bug report app crash: upload freeze",
        "bug report app crash: memory leak",
        "bug report app crash: submit button",
    ],
}


def render_ticket_text(topic_id: int, tenant_id: str, rng: np.random.Generator) -> str:
    """Render ticket body + a low-salience tenant marker as a trailing
    signature line (not the first line, per PLAN.md's tenant-salience risk)."""
    template = rng.choice(TOPIC_TEMPLATES[topic_id])
    return f"{template}\n\n— submitted via support portal (ref: {tenant_id})"


from dataclasses import dataclass


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    override: bool  # static regime: does this tenant's traffic use the mismatch permutation?


def build_tenants(n_tenants: int, alpha: float, seed: int) -> list[Tenant]:
    """alpha is the traffic-weighted fraction of tenants in override regime.
    Regime is static per tenant (not per-ticket) so it's learnable in-context
    across a tenant's history — this is what memory is supposed to exploit."""
    rng = np.random.default_rng(seed)
    tenants = []
    for i in range(n_tenants):
        tenant_id = f"T{i:04d}"
        override = bool(rng.random() < alpha)
        tenants.append(Tenant(tenant_id=tenant_id, override=override))
    return tenants


def resolve_correct_action(
    topic_id: int,
    tenant: Tenant,
    default_action_map: dict[int, str],
    mismatch_perm: dict[int, int],
) -> str:
    """The ground-truth label the policy/memory system is trying to predict."""
    if tenant.override:
        effective_topic = mismatch_perm[topic_id]
        return default_action_map[effective_topic]
    return default_action_map[topic_id]


from typing import Iterator


@dataclass(frozen=True)
class Ticket:
    step: int
    tenant_id: str
    topic_id: int
    text: str
    correct_action: str
    is_override: bool


class TicketGenerator:
    """Sequential ticket stream with a controllable similarity/utility
    mismatch (alpha) and an optional drift schedule (for Stage 5's H2:
    recurring/circular concept test)."""

    def __init__(
        self,
        alpha: float,
        n_tenants: int = 40,
        seed: int = 0,
        drift_period: int | None = None,
    ):
        self.alpha = alpha
        self.drift_period = drift_period
        self.default_action_map = build_default_action_map(seed=seed)
        self.tenants: dict[str, Tenant] = {
            t.tenant_id: t for t in build_tenants(n_tenants, alpha, seed=seed + 2)
        }
        # Each tenant gets its own independently-drawn derangement, rather
        # than sharing one global mismatch_perm. A single shared perm makes
        # default_action_map ∘ mismatch_perm a *fixed* bijection topic->action
        # for every override tenant, so similarity still fully predicts
        # action-agreement between any two override tenants' tickets. With
        # per-tenant perms, that composition differs tenant-to-tenant, which
        # decouples similarity from action-agreement across the population.
        # Built up front for every tenant (not just base-override ones) so a
        # tenant that flips into override via drift_period always has one
        # available (see _current_tenant).
        self.tenant_mismatch_perms: dict[str, dict[int, int]] = {
            tenant_id: build_mismatch_permutation(seed=seed + 1000 + i)
            for i, tenant_id in enumerate(self.tenants.keys())
        }
        self._rng = np.random.default_rng(seed + 3)
        self._tenant_ids = list(self.tenants.keys())

    @property
    def action_space(self) -> list[str]:
        return list(ACTION_LABELS)

    def _current_tenant(self, tenant_id: str, step: int) -> Tenant:
        base = self.tenants[tenant_id]
        if self.drift_period is None:
            return base
        flips = step // self.drift_period
        override = base.override if flips % 2 == 0 else not base.override
        return Tenant(tenant_id=base.tenant_id, override=override)

    def sample(self, step: int) -> Ticket:
        tenant_id = self._rng.choice(self._tenant_ids)
        topic_id = int(self._rng.integers(0, N_TOPICS))
        tenant = self._current_tenant(tenant_id, step)
        text = render_ticket_text(topic_id, tenant_id, self._rng)
        correct_action = resolve_correct_action(
            topic_id,
            tenant,
            self.default_action_map,
            self.tenant_mismatch_perms[tenant_id],
        )
        return Ticket(
            step=step,
            tenant_id=tenant_id,
            topic_id=topic_id,
            text=text,
            correct_action=correct_action,
            is_override=tenant.override,
        )

    def stream(self, n_steps: int) -> Iterator[Ticket]:
        for step in range(n_steps):
            yield self.sample(step)
