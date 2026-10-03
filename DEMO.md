# Five-Minute Demo Notes

This sequence uses only commands supported by the final code. TextOS reads memory and previews actions; it never executes an integration.

1. **Problem and approach — 20 seconds**
   - Explain that workplace facts change, evidence has delivery times, and answers need exact provenance.
   - Point to the two paths in the README architecture diagram: grounded memory and typed dry-run actions.

2. **Current versus historical memory — 60 seconds**

   ```bash
   python3 main.py textos --once "When is Route Planner v2 launching?" --as-of "2026-09-18T09:00:00-07:00"
   python3 main.py textos --once "When is Route Planner v2 launching?" --as-of "2026-09-15T12:00:00-07:00"
   ```

   - Highlight that `as_of` changes the visible evidence rather than post-filtering an answer.

3. **Provenance and abstention — 40 seconds**

   ```bash
   python3 main.py textos --once "What did Harbor Logistics say about SOC 2?" --as-of "2026-09-18T18:00:00-07:00"
   ```

   - Show that supported answers list exact source IDs and unsupported questions return `I don't know.`

4. **Dry-run actions — 60 seconds**

   ```bash
   python3 main.py textos --once "Message Sarah on Slack that the geocoding fix looks good" --as-of "2026-09-16T10:00:00-07:00"
   python3 main.py textos --once "Move board deck prep to 3pm" --as-of "2026-09-17T12:00:00-07:00"
   python3 main.py textos --once "Remind me an hour before the board meeting to print the deck" --as-of "2026-09-18T09:00:00-07:00"
   ```

   - Point out resolved IDs, offset-aware timestamps, preserved event duration, and the explicit dry-run notice.

5. **Safety gates — 30 seconds**

   ```bash
   python3 main.py textos --once "Delete all my emails from Marcus" --as-of "2026-09-18T09:00:00-07:00"
   python3 main.py textos --once "Message Sarah about the pricing proposal" --as-of "2026-09-18T09:00:00-07:00"
   ```

   - Show confirmation for destructive intent and clarification for unresolved identity ambiguity.

6. **Measured evaluation — 30 seconds**

   ```bash
   python3 scripts/verify_submission.py
   ```

   - Summarize the provided train-set results honestly: 100% primary retrieval, ~48% offline strict memory score with `judge=none` after the generic answering refactor (hard failures 0), and 12/12 action commands. Note that train accuracy is an evaluation metric, not a hidden-test claim.

7. **Close — 20 seconds**
   - Reiterate the design priorities: temporal correctness, exact provenance, deterministic offline reproduction, and zero real side effects.
