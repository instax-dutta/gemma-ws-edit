# GATES: Weight-space programming plan and docs update

OUTCOME: Update repo docs with baseline scores and weight-space programming plan; create actionable plan for editing weights without finetuning/retraining.

GATE: docs-updated
  DESC: Root AGENTS.md updated with new goal and baseline
  CHECK: grep -n "weight-space programming" AGENTS.md | wc -l
  EXPECT: grep -q "weight-space programming" AGENTS.md && echo ok

GATE: plan-doc-created
  DESC: Weight-space programming plan doc exists
  CHECK: test -f docs/WEIGHTSPACE_PROGRAMMING_PLAN.md && echo ok
  EXPECT: test -f docs/WEIGHTSPACE_PROGRAMMING_PLAN.md && echo ok

GATE: baseline-documented
  DESC: Baseline scores documented in plan
  CHECK: grep -n "GSM8K" docs/WEIGHTSPACE_PROGRAMMING_PLAN.md | wc -l
  EXPECT: grep -q "1.82\|0.0182" docs/WEIGHTSPACE_PROGRAMMING_PLAN.md && echo ok

GATE: compute-notes-documented
  DESC: Notes about laptop not feasible, use CPU server/VM documented
  CHECK: grep -n "CPU server\|not feasible" docs/WEIGHTSPACE_PROGRAMMING_PLAN.md | wc -l
  EXPECT: grep -q "CPU server\|VM\|not feasible" docs/WEIGHTSPACE_PROGRAMMING_PLAN.md && echo ok
