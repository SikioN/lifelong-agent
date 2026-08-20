"""Oracle: the paper's upper-bound baseline (arXiv-2605.10870v1,
neurips_2026.tex line 2790) -- "has access to the true decision identity".
Not a realistic memory method: write() is a no-op, and retrieve() uses
privileged access to the TicketGenerator's tenant/override state to
directly compute the ground-truth correct action for the current ticket,
handing it to the policy as a stated fact rather than a learned precedent.
"""
from env.generator import Ticket, TicketGenerator, resolve_correct_action


class OracleMemory:
    """Does not subclass BaseMemory: BaseMemory's write/retrieve contract
    is built around learning from past MemorySlots, and Oracle explicitly
    doesn't -- it needs a reference to the TicketGenerator instead, which
    no other memory method receives (and shouldn't, since a real agent
    can't observe it). Still implements the same write(ticket, action,
    correct)/retrieve(ticket) -> str shape so Stage 4/5's experiment
    runner can treat it uniformly with the BaseMemory subclasses."""

    def __init__(self, budget: int, generator: TicketGenerator):
        self.budget = budget
        self.generator = generator

    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        pass

    def retrieve(self, ticket: Ticket) -> str:
        tenant = self.generator._current_tenant(ticket.tenant_id, ticket.step)
        correct_action = resolve_correct_action(
            ticket.topic_id,
            tenant,
            self.generator.default_action_map,
            self.generator.tenant_mismatch_perms[ticket.tenant_id],
        )
        return f"Oracle precedent: the correct action for this exact situation is {correct_action}."
