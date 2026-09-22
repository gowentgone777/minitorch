from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import deque
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg]+=epsilon
    vals2[arg]-=epsilon
    
    return (f(*vals1) - f(*vals2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sort_node = deque()
    visited = set()
    def visit(v : Variable):
        if v.unique_id in visited or v.is_constant():
            return
        if not v.is_leaf():
            for parent in v.parents:
                visit(parent)

        visited.add(v.unique_id)
        sort_node.appendleft(v)
        
    visit(variable)
    return sort_node



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sort_node = topological_sort(variable)
    props = {variable.unique_id :deriv}
    props[variable.unique_id] = deriv
    for parent in sort_node:
        d_output = props[parent.unique_id]
        if parent.is_leaf():
            parent.accumulate_derivative(d_output)
        else:
            for node, d_node in parent.chain_rule(d_output):
                if node.is_constant():
                    continue
                if node.unique_id not in props:
                    props[node.unique_id] = d_node
                else:
                    props[node.unique_id] += d_node



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
