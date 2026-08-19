from agent.prompt_templates import build_prompt
from env.generator import ACTION_LABELS, TicketGenerator
gen = TicketGenerator(alpha=0.0, seed=777)
ticket = gen.sample(0)
prompt = build_prompt(ticket.text, ACTION_LABELS, "RULE CONTEXT")
print(prompt)
