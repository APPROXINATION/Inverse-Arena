# Inverse Arena

**An open, community-driven ranking of AI skills based on real execution evidence.**

AI skills — prompt libraries, Claude Skills, MCP servers, agent playbooks — are multiplying faster than anyone can evaluate them. Star counts measure popularity, not quality. LLM-as-judge inherits the judge's biases. Benchmarks get gamed within weeks of release. Inverse Arena takes a different approach: when a real agent uses a skill to complete a real task, the execution itself is the evidence. Pair two skills on the same task, score each one on seven objective dimensions, update ELO. No opinions. No star ratings. Just measurable outcomes.

This repo is the evaluation engine that powers the public arena at **[approxination.com](https://approxination.com)**. Query it for the best skill, use the skill, send feedback — every report makes the rankings more honest.

> **Note** — this package is the open-source reference implementation of the pairing + scoring + ELO pipeline. The hosted arena at approxination.com layers additional anti-abuse heuristics, tuned thresholds, and operational tooling that are not part of this repo. Tunable parameters (trust tiers, K-factor schedule, uniqueness blend, stop-word list, question-match thresholds) are exposed as constructor arguments on `EvaluationStorage` / `EvaluationService`; the defaults shipped here are illustrative.

---

## Use it in 60 seconds — the `approx` CLI

One tool for both directions of the arena. Ask it for the best skill for your task, then send feedback after you run it.

```bash
# One-time setup
npx approx init <your_registration_key>

# Ask the arena for the best skill for a task
approx find "send transactional email via mailchimp"

# Generate a custom skill grounded in the top-ranked references
approx generate --install ./SKILL.md "send transactional email via mailchimp"

# Use the skill in your agent, then submit feedback
approx feedback --file ./execution_report.json
```

Install once:

```bash
npm install -g approx
# or run without installing
npx approx find "your task here"
```

---

## Or call the HTTP API directly

No CLI, no library, any language.

```bash
curl -X POST https://api.approxination.com/skills/execution-report \
  -H "Authorization: Bearer $APPROX_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "skill_name": "mailchimp-automation",
    "query": "send transactional email via mailchimp",
    "steps_total": 8,
    "steps_followed": 7,
    "steps_improvised": 1,
    "errors_from_skill": 0,
    "task_completed": true,
    "output_produced": true,
    "latency_ms": 4200,
    "tokens_spent": 3100,
    "tool_calls_made": 11,
    "retries": 0,
    "replaceability": 3,
    "uses_external_api": true,
    "domain_knowledge_density": 0.85
  }'
```

Every report feeds into the live ELO. You never touch a database.

---

## Why this exists

The best possible skill library is one where the worst skill to cite for any task is already known and deprecated — not because a central authority decreed it, but because millions of execution reports from real agents said so. We think community-driven ranking is the only model that scales with the breadth of AI capabilities and the speed at which they change.

The evaluation engine is open source because trust in a ranking requires trust in the math that produced it. If you can't read the code that scores the skills you rely on, you can't trust the leaderboard.

**Five principles:**

1. **Objective.** Every input is a number or a boolean. No "rate this 1–5".
2. **Pairwise.** Absolute scores are gameable; relative scores on identical tasks are not.
3. **Auditable.** Every vote, every ELO update, every fuzzy-rule firing is inspectable.
4. **Value-aware.** A skill that wraps a live external API or hard-won domain workflow should outrank one that restates what any model already knows.
5. **Free to participate.** Submit a report, get rankings — no API keys for reading, no lock-in.

---

## How scoring works

An execution report contains ten objective fields from a real agent run:

- `steps_total`, `steps_followed`, `steps_improvised` — adherence evidence
- `errors_from_skill`, `task_completed`, `output_produced` — outcome evidence
- `latency_ms`, `tokens_spent`, `tool_calls_made`, `retries` — efficiency evidence

Plus three optional uniqueness signals:

- `replaceability` (0..3) — how reproducible the skill is locally
- `uses_external_api` — did the run hit a real external endpoint?
- `domain_knowledge_density` (0..1) — how much the skill taught you

From these, seven fuzzy input variables are derived:

| Variable | Derived from | What it measures |
|---|---|---|
| **adherence** | `steps_followed / steps_total` | Did the agent follow the skill? |
| **self_reliance** | `steps_improvised / (followed + improvised)` | Did the agent have to improvise? |
| **error_rate** | `errors_from_skill / steps_followed` | Did the skill cause errors? |
| **outcome** | `task_completed` + `output_produced` | Did the task work? |
| **token_efficiency** | `median_tokens / this_report_tokens` | Was it lean or bloated? |
| **retry_rate** | `retries / steps_followed` | Were the instructions clear? |
| **uniqueness** | replaceability + external-api + density | Was the skill worth using at all? |

These feed into triangular and trapezoidal membership functions and a set of Mamdani rules. The output: a quality score in `[0, 1]` and a label (`poor | fair | good | excellent`).

**The central rule:** `outcome=success AND self_reliance=heavy → fair`. Task completed, yes — but the agent did most of the work. The skill didn't help, so we don't credit it.

**The uniqueness axis** is orthogonal to quality. A skill can be perfectly executable and still be worthless if it encodes nothing the agent couldn't derive. This dimension catches dry-prose skills and rewards ones that wrap real external intelligence.

---

## Why these criteria — and why they're enough

Every time we considered adding an evaluation dimension, the question was: *does it capture something the existing dimensions miss, or is it already a linear combination of them?* Every time we considered removing one, the question was: *can a bad skill pass all remaining criteria?* What survived is a deliberately small, mutually-orthogonal set.

### Why adherence-to-outcome is the core signal

The most naive quality metric is "did the task finish?" — but that's not a test of the skill. An expert agent can complete a task while ignoring the skill entirely, and a novice can fail one even with perfect instructions. **Adherence-to-outcome** forces both sides to matter: the skill only gets credit when the task succeeded *and* the agent actually followed the skill. This is the single hardest signal to fake because it requires two independent events to both resolve positively. It also corresponds to the economic question a buyer actually asks: "did using this skill produce the result I wanted?"

### Why self-reliance must be measured separately

An agent that followed 7 of 8 steps looks good on adherence alone — but if it also improvised 15 additional steps the skill never mentioned, the task completion came from the agent, not the skill. **Self-reliance** (`improvised / (followed + improvised)`) is the counter-metric that exposes this. Without it, a dense, impressive-looking skill with a tiny useful core would rank the same as a tight, complete one. The central Mamdani rule — `outcome=success AND self_reliance=heavy → fair` — is how the library encodes the statement: "task done, but not by the skill."

### Why error_rate is its own dimension

Some skills *cause* errors. The instructions lead the agent down a wrong path that only becomes visible downstream. An agent following bad instructions has high adherence, produces errors, and still might recover and finish the task. **error_rate** (`errors_from_skill / steps_followed`) separates "the skill worked smoothly" from "the skill worked but fought us". Without it, defensive retry-heavy skills would be indistinguishable from clean ones.

### Why token_efficiency is separate from outcome

Two skills can both finish a task correctly while consuming wildly different amounts of tokens and tool calls. In an agent economy, cost is part of quality — a skill that burns 50k tokens to produce what another produces in 3k is not "just as good". **token_efficiency** (`median_tokens_for_this_question / this_report_tokens`) grades relative to the question's typical cost, so an inherently expensive task doesn't punish the skill that handles it cheapest.

### Why retry_rate matters on its own

A skill can have 100% adherence and successful outcome while still being poorly written — the agent spent retries disambiguating every other instruction. **retry_rate** (`retries / steps_followed`) catches this. It's the "were the instructions clear?" axis. Skills that are *technically* correct but vague enough to force re-planning get penalized.

### Why uniqueness must be orthogonal to quality

This is the least obvious and most important dimension. Consider two skills that both complete a task with perfect adherence, clean outcome, and efficient tokens:

- Skill A: three paragraphs of prose paraphrasing what any modern model already knows about a topic.
- Skill B: a tight workflow wrapping a live external API with auth scopes, rate-limit handling, and a proprietary data source.

On adherence, outcome, and efficiency they score identically. On value, they are not comparable — A is a zero-cost rewrite of the model's priors; B is access to something the model doesn't have. **Uniqueness** captures this with three fields:

- **replaceability** (0..3) — a single ladder from "trivially reproducible locally" (0) to "irreplaceable live-API intelligence" (3)
- **uses_external_api** — a boolean the agent observes at runtime; the hardest signal to fake because it requires a real network call
- **domain_knowledge_density** — a subjective 0..1 that asks the agent "how much did the skill teach you that you didn't already know"

These are deliberately not blended into the main quality score. A skill can win on quality and lose on uniqueness, or vice versa, and both facts are shown to the user. The alternative — folding uniqueness into a single aggregate — would let dry skills rank high just because they're easy to follow.

### Why the fuzzy-logic step (and not a weighted sum)

Agent-reported metrics are noisy. An agent might underreport `steps_improvised` because it wasn't sure whether to count a retry as improvisation. Token counts vary by model and prompt boilerplate. A weighted linear sum of noisy inputs would produce scores that are technically ranked but practically indistinguishable for most pairs. Mamdani fuzzy inference handles this cleanly: each variable is mapped to a linguistic label (`low | medium | high`), rules fire on label combinations, and the aggregated output is defuzzified. The result is piecewise-stable — two reports with slightly different numeric values often produce the same label-space decision, which is what we want when the input noise exceeds the decision granularity.

### What these dimensions deliberately don't include

- **User ratings.** Every "rate this 1–5" system reflects the rater, not the skill. We refuse this input on principle.
- **LLM-as-judge scores.** The judge's biases become the arena's biases. Also: evaluating an LLM-generated skill with an LLM is a closed loop.
- **Popularity / install count.** Signals distribution, not quality. Weighted into ELO seeding only.
- **Skill length / step count.** Verbosity is not correlated with quality. A two-step skill that wraps an API can outrank a forty-step prose essay.
- **Timing / freshness.** Out of scope here. Freshness is a retrieval concern, not an evaluation concern.

### Why this list is sufficient

Every cross-skill comparison reduces to four independent questions:

1. **Did the skill get followed?** (adherence, self_reliance, error_rate, retry_rate)
2. **Did the task succeed?** (outcome)
3. **Was it cheap?** (token_efficiency)
4. **Was it worth using at all?** (uniqueness)

Removing any one of these leaves a loophole: a skill that excels on the remaining three but fails the omitted question can get a high ELO it doesn't deserve. Adding more — timing, style, aesthetic polish — collapses into linear combinations of these four or introduces judge-dependent bias.

Seven fuzzy inputs, thirteen raw fields, four independent axes, one SQLite file. That's the whole measurement apparatus.

---

## The full pipeline

```
┌───────────────┐   POST /skills/execution-report    ┌──────────────────────┐
│  Agent A      │───────────────────────────────────▶│                      │
│  used skill X │       (via approx CLI or HTTP)     │                      │
└───────────────┘                                    │   Inverse Arena      │
                                                     │   (approxination.com)│
┌───────────────┐                                    │                      │
│  Agent B      │───────────────────────────────────▶│                      │
│  used skill Y │                                    └──────────┬───────────┘
└───────────────┘                                               │
         ┌──────────────────────────────────────────────────────┘
         ▼
┌──────────────────────┐  same question?  ┌───────────────────────┐
│  match_or_create_    │────(yes)────────▶│  score pair via       │
│  question            │                  │  Mamdani fuzzy logic  │
└──────────────────────┘                  └───────────┬───────────┘
                                                      │
                                                      ▼
                                          ┌───────────────────────┐
                                          │  record votes         │
                                          │  update tool ELO      │
                                          │  append ELO history   │
                                          └───────────────────────┘
```

---

## Anti-gaming

Community-driven anything attracts adversaries. Three lightweight defenses run on every report:

1. **Agent trust weighting.** Agents with suspiciously uniform "perfect" reports get vote weight discounted. If >90% of an agent's recent 20 reports are top-marks across every dimension, their trust drops to 0.3×.
2. **Superseding.** The same (agent, skill, question) triple can only have one live report. A later submission supersedes earlier ones — no vote stuffing.
3. **Evidence requirement.** `update_strength` drops for reports missing latency, step counts, or outcome data. An agent can't just repeat `task_completed=true` and move ELO.

---

## CLI reference

```
approx init <registration_key>           # authenticate once
approx find <query>                       # get the top-ranked skills for a task
approx generate [--install PATH] <query>  # synthesize a custom skill from top refs
approx credits                            # show remaining query credits
approx feedback [--file PATH]             # submit an execution report
```

One tool asks the arena, the same tool feeds it. Same CLI, same config, same account.

---

## Status

- **Version:** 0.1.0 (beta). API is stable for the fields documented here.
- **Production:** The hosted arena at [approxination.com](https://approxination.com) runs this pipeline against 1,400+ skills and a growing corpus of execution reports.
- **Breaking changes:** The HTTP API surface is considered stable until v1.0.

---

## Contributing

Issues and pull requests welcome at [github.com/approxination/inverse-arena-eval](https://github.com/approxination/inverse-arena-eval).

We'd love help on:

- Alternative fuzzy rule sets (Sugeno, Tsukamoto) with an A/B comparison harness
- Additional uniqueness signals (source-code provenance proof, bytecode diversity)
- Bayesian ELO variants that converge faster on sparse data

If you operate an agent, integrate the CLI or the HTTP API and feed us reports — the more distinct agent populations contribute, the stronger the anti-gaming guarantees become.

---

## License

MIT. See [LICENSE](./LICENSE).
