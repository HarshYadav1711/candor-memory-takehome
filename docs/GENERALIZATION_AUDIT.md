# Generalization Audit

## Scope and guardrails

This is a Phase 7A measurement artifact. It audits the submitted runtime at commit `d70a0db8f9fb7614412b3c109dd115da56232cda` and adds independent development tests. It does not change retrieval, answering, action planning, visibility, safety, ingestion, or TextOS behavior.

The hidden questions and commands were not available and were not reconstructed. The suites were written from the source dataset, ordinary English paraphrase families, and deliberately absent slots. Expected fields remain in test data; each evaluator passes only `id`, `question`/`command`, and `as_of` to runtime.

## Answer-layer brittleness

No runtime branch literally hard-codes a train question ID, evaluator annotation, or gold answer. The main risk is train-shaped selection logic: generic in intent, but narrow in how a question must be phrased.

| Location | Finding | Generalization risk |
|---|---|---|
| `answering/extractive.py:536` `write_extractive_answer` | Fourteen selectors run in a fixed precedence order, followed by one generic fallback. | A new wording can miss its intended family or trigger the wrong earlier family even when retrieval is good. |
| `answering/extractive.py:127` `_temporal_difference` | Requires the shape `how many days after ... did ...`. | Equivalent before/after or elapsed-time wording bypasses temporal composition. |
| `answering/extractive.py:156` `_corrected_metric` and `:173` `_passing_count` | Depend on `pNN` and `how many` lexical forms plus narrow value regexes. | Correct evidence is not enough when a metric is requested indirectly. |
| `answering/extractive.py:190` `_flight_lookup` | Activates only when the question contains `flight` and expects one email-like layout. | “Outbound to Denver” or a different itinerary format falls through. |
| `answering/extractive.py:260` `_ownership_status` | Requires `who ... owns/doing`; owner extraction looks for a small evidence phrase list. | Possessive, passive, or “responsible for” questions do not use the ownership path. |
| `answering/extractive.py:284` `_conditional_answer` | Question gating relies on a small conditional vocabulary or starts with `are/is/will/do`. | Proposed, contingent, tentative, and prerequisite phrasing can lose modality. |
| `answering/extractive.py:342` `_choice_reason` | Requires `which/what`, a `pick/choose/chose/select` form, and `why`. | “Recall the storage decision and rationale” misses the choice composer. |
| `answering/extractive.py:391` `_prediction_disagreement` | Recognizes only `going to/will/expect` near `sign/close`. | Disputes about other outcomes or differently phrased forecasts are not preserved. |
| `answering/extractive.py:408` `_multi_part_request` | Requires the literal ordering `which ... and ... what`. | Other two-slot questions often return one sentence and omit one requested value. |
| `answering/extractive.py:425` `_attribution_or_disagreement` | Direct-vs-reported handling depends on speaker hints and a small `said/told/reported` pattern. | Passive voice and indirect attribution can select unrelated speakers or flatten provenance. |
| `answering/extractive.py:486` `_commitment_status` | Activates from a narrow intent list or questions beginning `did/has/do/does`; the first marker-ranked resolved sentence wins. | “Was it delivered?” and unfamiliar completion/cancellation verbs can miss or misclassify state. |
| `answering/extractive.py:523` `_generic_answer` | Returns the highest-scoring sentence whenever any candidates remain. | Topical evidence is treated as an answer even when the requested slot is absent. This is the clearest source of false non-abstention. |
| `answering/support.py:27` `_focus_terms` | Extracts a requested focus only from final `about ...` or one possessive `what is X's Y` shape. | “Tell me Dana's salary” has no recognized focus; evidence about Dana can pass the gate without salary support. |
| `answering/support.py:39` `decide_support` | Uses lexical coverage, source/entity/date signals, and evidence counts rather than proving a requested slot/value relation. | Entity overlap can authorize an off-topic answer. The holdout produced 0% abstention accuracy across 20 negative and near-miss cases. |
| `answering/engine.py:35` `GroundedMemory.answer` | Only `INSUFFICIENT_EVIDENCE` abstains; `AMBIGUOUS` proceeds to writing. | A comparison signal is descriptive, not a fail-closed state, and may still compose one-sided evidence. |
| `retrieval/planner.py:271`, `:293`, `:303` | Anchor, speaker, and intent detection use small phrase/verb regex sets. | Unseen syntax loses reranking signals before answer composition begins. |
| `retrieval/hierarchical.py:268` `_rank_visible` | BM25 plus lexical coverage, metadata, dates, entities, and fixed intent boosts. | Safe and inspectable, but synonym and relation recall remain close to keyword search when question vocabulary differs from source text. |

Legitimate broad rules should be retained: visibility is applied before retrieval (`hierarchical.py:238`), retrieval indexes sanitized text, BM25 is corpus-generic, sources are restricted to visible retrieved units, and model output is validated. The issue is not that these rules exist; it is that answer authorization and composition depend too heavily on a few surface forms.

## Action-layer brittleness

| Location | Finding | Generalization risk |
|---|---|---|
| `actions/parser.py:64` `_ATOMIC_START` and `:73` `decompose_command` | Conjunction splitting recognizes only a fixed imperative list. | Compound paraphrases such as “send ... and ping ...” remain one clause and usually clarify. |
| `actions/planner.py:85` `_plan_atomic` | Intent dispatch is a start-of-string regex over `open`, `remind`, `move/reschedule`, `book/schedule`, `email`, and `message/tell/thank`. | Common verbs such as launch, bring up, ping, notify, write, arrange, create, shift, and change are unseen. |
| `actions/planner.py:115` `_person_from_clause` | Recipient extraction accepts only `email/message/thank` before a single token. | `tell`, `send`, `write to`, and multiword constructions cannot resolve a person even if the contact directory can. |
| `actions/planner.py:140` `_message` | Channel resolution is attached specifically to `tell`; person messaging expects `that/about/to say`. | Ordinary Slack paraphrases fail or lose message content. The holdout Slack category scored 0/8. |
| `actions/planner.py:180` `_gmail` | The email path requires the command to begin `email`. | Seven of eight email paraphrases failed despite resolvable contacts. |
| `actions/planner.py:209` `_calendar_create` | Requires `book/schedule`, explicit duration/date/time, and a `with PERSON` ordering rewritten into an email-shaped string. | Equivalent scheduling syntax fails before entity/time resolution. |
| `actions/planner.py:235` `_calendar_update` | Requires `move/reschedule EVENT to TIME`. | Push back, bring forward, shift, and change-time forms are not mapped. |
| `actions/planner.py:251` `_reminder` | Requires the first word `remind`; event-relative parsing assumes a specific `OFFSET before EVENT to TEXT` shape. | “Don't let me forget”, “give me a reminder”, and “ping me before” fail. |
| `actions/temporal.py:67` `resolve_time` | Bare hours 1–6 are assumed to be PM. | Convenient for train commands, but ambiguous natural-language times can be silently resolved instead of clarified. |
| `actions/planner.py:301` `_match_event` | Uses token overlap, a 0.5 threshold, and exact-score tie rejection. | Generic phrases can choose one of several planning/board events rather than clarify. |
| `actions/entities.py:198` `resolve_person` | Topic overlap can auto-select one same-first-name contact when the score gap is two terms. | Helpful when context is strong, but it can turn vague wording into a speculative recipient. |
| `actions/safety.py:8` `_DESTRUCTIVE` and `:18` `needs_confirmation` | Detection covers seven verbs before a small object list; cancel, trash, discard, and “get rid of” are absent. | Supported destructive paraphrases return clarification rather than confirmation. Confirmation accuracy was 8/12, although no direct side-effect action escaped. |
| `actions/safety.py:13` `_CONFIRMED` | A few phrases bypass the confirmation gate before intent parsing. | Confirmation state is inferred from command text rather than an explicit two-step interaction state. |
| `actions/planner.py:55` `plan` | Safety runs before intent parsing, and any clarification discards the entire compound plan. | Pre-parse safety is good and should remain; fail-closed compound handling is safe but hides which independent clauses were understood. |

Broadly generic pieces worth retaining are the data-derived contact directory, timezone-aware temporal utilities, typed action allowlist/validation, dry-run-only planner boundary, and safety gate before planning.

## Train-set tuning vs general behavior

### Broadly generic

- Canonical ingestion, immutable IDs, temporal projection, Slack edit/deletion handling, and sanitation.
- BM25 indexing, record/unit hierarchy, visibility-first retrieval, exact source validation, and deterministic bounds.
- Dataset-derived contacts, channels, emails, calendar events, and typed dry-run actions.
- Fail-closed clarification and the pre-planning destructive gate.

### Added while iterating against train failures

- Current-state, causal, disagreement, attribution, and temporal-anchor retrieval boosts.
- Answer selectors for launch-like state changes, corrected metrics, flight lookup, ownership/completion, pricing, dictated email, disagreement, and commitment status.
- Action verb dispatch and command-specific content grounding used by the twelve provided commands.

These components are not literal hard-coding. They are train-shaped heuristics: each encodes a legitimate semantic family, but often through the exact lexical and syntactic forms exercised during train iteration.

### Likely to overfit and candidates for replacement in Phase 7B/7C

- Question-shape selector activation and fixed selector precedence.
- Lexical support as a substitute for requested-slot support.
- Rank-first generic answering after a permissive support decision.
- Start-verb action dispatch and one-word-order argument extraction.
- Destructive intent represented only by a verb/object regex.

The visibility, safety ordering, source constraints, deterministic interfaces, and typed action schemas should be retained around any replacement.

## Independent holdout design

### Memory suite

`evals/memory_generalization.jsonl` contains 58 cases:

| Category | Count |
|---|---:|
| unseen record | 16 |
| unseen-question-form paraphrase | 10 |
| absence/abstention | 10 |
| near-miss abstention | 10 |
| temporal | 4 |
| attribution | 2 |
| disagreement | 2 |
| conditional/modality | 2 |
| multi-hop | 2 |

Twenty cases are explicitly tagged `unseen_record_holdout`. They use records not listed in any official train `needed` group: 34.5% of all memory cases and 52.6% of the 38 supported-answer cases. The remaining supported cases deliberately stress new wording, temporal state, attribution, disagreement, and composition. Negative cases ask for absent slots rather than inventing new facts.

The suite does not perform one-word substitutions of train questions. Transformations include statement-like requests, “remind me”, “do we know”, indirect state descriptions, passive/direct attribution contrasts, different clause order, and new cross-source combinations. Test-only `expected`, `tags`, and `evidence` fields document provenance and are never passed to runtime.

### Action suite

`evals/actions_generalization.jsonl` contains 80 cases:

| Category | Count |
|---|---:|
| Slack paraphrases | 8 |
| email paraphrases | 8 |
| calendar create paraphrases | 8 |
| calendar update paraphrases | 8 |
| reminder paraphrases | 8 |
| memory-query paraphrases | 6 |
| app-open paraphrases | 5 |
| ambiguity | 7 |
| destructive confirmation | 12 |
| destructive negative controls | 4 |
| multi-action | 6 |

Commands were written by semantic family, not by copying the twelve official commands and replacing names. Destructive cases cover delete, remove, erase, wipe, clear, trash, purge, get rid of, cancel, discard, and permanent removal. Negative controls distinguish deleting an object from editing attendee or draft content.

## Frozen-runtime baseline

### Memory

| Metric | Result |
|---|---:|
| Overall | 22.41% (13/58) |
| Supported answers | 34.21% (13/38) |
| Abstention | 0.00% (0/20) |
| Forbidden-term accuracy | 100.00% |

| Category | Accuracy |
|---|---:|
| unseen record | 25.00% |
| paraphrase | 50.00% |
| abstention | 0.00% |
| near-miss abstention | 0.00% |
| temporal | 100.00% |
| attribution | 0.00% |
| disagreement | 0.00% |
| conditional | 0.00% |
| multi-hop | 0.00% |

### Actions

| Metric | Result |
|---|---:|
| Overall exact plan | 37.50% (30/80) |
| Exact action-type plan | 37.50% |
| Argument resolution | 22.22% |
| Destructive confirmation | 66.67% (8/12) |
| No direct side-effect action for destructive/negative controls | 100.00% |
| Ambiguity | 57.14% (4/7) |
| Multi-action count/type | 50.00% (3/6) |

| Paraphrase category | Accuracy |
|---|---:|
| Slack | 0.00% |
| email | 12.50% |
| calendar create | 25.00% |
| calendar update | 37.50% |
| reminder | 12.50% |
| memory query | 66.67% |
| app open | 20.00% |

## Common general failure classes

1. Topically related evidence passes the support gate even when the requested slot is absent; all 20 abstention/near-miss cases answered.
2. A supported answer is selected from the wrong record or only one requested part is composed; unseen-record, attribution, disagreement, conditional, and multi-hop categories expose this.
3. Intent recognition is bound to command-initial train verbs; 8/8 Slack paraphrases and 7/8 email paraphrases failed.
4. Argument parsing assumes one word order after intent dispatch, producing low argument-resolution accuracy.
5. Destructive safety fails closed but does not consistently distinguish “confirmation required” from unsupported clarification, and one negative control is over-classified as destructive.
6. Ambiguous contacts/events can be resolved speculatively by context or token overlap instead of always clarifying.

Detailed failures are regenerated at:

- `outputs/generalization_memory_failures.jsonl`
- `outputs/generalization_action_failures.jsonl`

## Limitations

- This is a hand-authored development holdout over one small fictional corpus, not a statistically independent external benchmark.
- Substring expectations intentionally test essential values rather than prose quality; they can miss semantically equivalent wording not listed as an alternative.
- Negative cases were checked against the supplied visible corpus, but absence is always easier to establish for concrete slots than broad open-ended questions.
- The suite measures the frozen deterministic extractive path and dry-run planner only; it does not use the optional model writer.
- No Phase 7B answer refactor or Phase 7C action refactor is included here.
