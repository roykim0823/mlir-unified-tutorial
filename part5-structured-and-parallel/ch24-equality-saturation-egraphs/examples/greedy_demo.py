"""The phase-ordering problem, demonstrated with a *destructive* rewriter.

This is the strategy of MLIR's greedy pattern driver (Chapters 10, 16, 17):
find a match, replace it, repeat until nothing matches. Each rewrite throws
the old form away, so the result depends on *which rule fires first* — on
pass order, and even, inside one driver, on the order the IR is visited.
We use the same two rules as optimize_demo.py, on (a*2)/2:

    strength reduction   x*2     -> x<<1
    cancellation         (x*y)/y -> x
"""

from optimize_demo import fmt


def strength_reduce(t):
  if t[0] == "*" and t[2] == ("lit", 2):
    return ("shl", t[1], ("lit", 1))
  return None


def cancel(t):
  if t[0] == "/" and t[1][0] == "*" and t[1][2] == t[2]:
    return t[1][1]
  return None


def rewrite_once(t, rules, top_down):
  """Apply the first matching rule at the first matching position, visiting
  the root before its children (top_down) or the children first."""
  def at_root():
    for rule in rules:
      new = rule(t)
      if new is not None:
        return new
    return None

  def in_children():
    if t[0] in ("lit", "sym"):
      return None
    for i, child in enumerate(t[1:], start=1):
      new = rewrite_once(child, rules, top_down)
      if new is not None:
        return t[:i] + (new,) + t[i + 1:]
    return None

  first, second = (at_root, in_children) if top_down else (in_children, at_root)
  new = first()
  return new if new is not None else second()


def greedy(t, rules, top_down=True):
  """Rewrite destructively until no rule matches anywhere (a fixed point)."""
  while (new := rewrite_once(t, rules, top_down)) is not None:
    t = new
  return t


def main():
  term = ("/", ("*", ("sym", "a"), ("lit", 2)), ("lit", 2))
  print(f"Greedy, destructive rewriting of {fmt(term)}:")
  print("  as two passes (each run to a fixed point, in sequence):")
  t = greedy(greedy(term, [cancel]), [strength_reduce])
  print(f"    cancel; then strength-reduce   ->  {fmt(t)}")
  t = greedy(greedy(term, [strength_reduce]), [cancel])
  print(f"    strength-reduce; then cancel   ->  {fmt(t)}   (stuck)")
  print("  as one driver holding both rules:")
  print(f"    visiting parents first         ->  {fmt(greedy(term, [cancel, strength_reduce], True))}")
  print(f"    visiting operands first        ->  {fmt(greedy(term, [cancel, strength_reduce], False))}   (stuck)")


if __name__ == "__main__":
  main()
