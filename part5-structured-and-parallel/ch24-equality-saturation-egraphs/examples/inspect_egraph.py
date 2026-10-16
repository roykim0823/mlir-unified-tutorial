"""Look inside the e-graph: print every e-class before and after saturating the
phase-ordering example (a*2)/2 with the two rules

    (x*y)/y -> x        and        x*2 -> x<<1

so you can see that nothing is ever deleted — classes only gain e-nodes and
get merged."""

from egraph import EGraph, Var
from optimize_demo import fmt


def v(name):
  return Var(name)


def main():
  eg = EGraph()
  root = eg.add(("/", ("*", ("sym", "a"), ("lit", 2)), ("lit", 2)))
  print(f"after add: root = c{root}")
  print(eg.dump())

  rules = [
    (("/", ("*", v("x"), v("y")), v("y")), v("x")),            # (x*y)/y -> x
    (("*", v("x"), ("lit", 2)), ("shl", v("x"), ("lit", 1))),  # x*2 -> x<<1
  ]
  eg.saturate(rules)
  print(f"\nafter saturate: root = c{eg.find(root)}")
  print(eg.dump())

  costs = {"lit": 0, "sym": 0, "shl": 1, "*": 1, "/": 5}
  print(f"\nextract(root) = {fmt(eg.extract(root, costs))}")


if __name__ == "__main__":
  main()
