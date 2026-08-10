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
        "I was charged twice for the same order and need this fixed.",
        "There's a charge on my statement I don't recognize.",
        "My invoice total doesn't match what I was quoted.",
        "I noticed a duplicate transaction on my card this week.",
    ],
    1: [  # password_reset
        "I can't log into my account, the password reset link isn't arriving.",
        "My login keeps failing even after I reset my password.",
        "I'm locked out of my account and need help getting back in.",
        "The reset email for my password never showed up.",
    ],
    2: [  # data_export_request
        "I need a full export of my account data for my records.",
        "Can you send me a copy of all the data stored under my account?",
        "I'd like to download my complete usage history.",
        "Please provide an export of everything associated with my account.",
    ],
    3: [  # account_deletion
        "I want to permanently close my account and remove my data.",
        "Please delete my account, I no longer want to use the service.",
        "I'd like to cancel and have my account fully removed.",
        "Can you shut down my account and erase my information?",
    ],
    4: [  # refund_request
        "I'd like a refund for a purchase that didn't work as expected.",
        "The item I bought was defective, I want my money back.",
        "Please refund my last payment, I was not satisfied.",
        "I'm requesting a refund since the service didn't meet expectations.",
    ],
    5: [  # shipping_delay
        "My order was supposed to arrive last week and still hasn't shown up.",
        "The tracking hasn't updated in days and my package is late.",
        "I'm still waiting on a delivery that's well past the estimated date.",
        "My shipment seems stuck somewhere and hasn't moved.",
    ],
    6: [  # feature_request
        "It would be great if the app supported dark mode.",
        "Could you add the ability to export reports as PDF?",
        "I'd love to see keyboard shortcuts added to the editor.",
        "Please consider adding a bulk-edit option to the dashboard.",
    ],
    7: [  # bug_report
        "The app crashes every time I try to open the settings page.",
        "I found a bug where the totals don't add up correctly.",
        "The page freezes whenever I upload a large file.",
        "There's a glitch that logs me out randomly during use.",
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
