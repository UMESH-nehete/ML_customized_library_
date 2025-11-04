# visualize.py

import graphviz
from my_ml_lib.nn.autograd import Value # Assuming Value is here
import numpy as np


# --- TODO: Implement Graph Traversal ---
def get_all_nodes_and_edges(root_node: Value):
    """
    Performs a backward traversal  from the root_node
    to find all unique Value nodes and the directed edges connecting them
    in the computation graph.

    Args:
        root_node (Value): The final node in the graph (e.g., the loss Value object).

    Returns:
        tuple: (nodes, edges)
               nodes (set): A set containing all Value objects found during traversal.
               edges (set): A set of tuples (parent_Value, child_Value) representing
                            the directed edges: parent -> child.
    """
    # --- TODO: Step 1 - Initialize Sets ---
    # Create an empty set `nodes` to store unique Value objects.
    # Create an empty set `edges` to store unique edge tuples (parent, child).
    # Create an empty set `visited` to keep track of nodes already processed.
    print("TODO: Initialize sets for nodes, edges, and visited in get_all_nodes_and_edges.")


    # --- TODO: Step 2 - Implement DFS Traversal Function (`build_graph`) ---
    # Define a recursive helper function (e.g., `build_graph(v)`) that takes a Value `v`.
    # Inside the helper function:
    #   a) Check if `v` has already been visited. If yes, return.
    #   b) Add `v` to the `visited` set.
    #   c) Add `v` to the `nodes` set.
    #   d) Iterate through the parents of `v` (e.g., `for parent in v._prev:`).
    #       i) Add the edge `(parent, v)` to the `edges` set.
    #       ii) Recursively call the helper function on the `parent`.
    print("TODO: Implement the recursive graph traversal logic in get_all_nodes_and_edges.")
    def build_graph(v):
        # Placeholder for the recursive logic
        pass

    # --- TODO: Step 3 - Start Traversal ---
    # Call your recursive helper function starting from the `root_node`.
    print("TODO: Start the traversal from the root_node in get_all_nodes_and_edges.")
    # build_graph(root_node) # Placeholder call

    # --- TODO: Step 4 - Return Results ---
    # Return the populated `nodes` and `edges` sets.
    return nodes, edges
# --- End TODO ---


# ---  Graph Drawing Function ---
def draw_dot(root_node: Value, format='svg', rankdir='LR'):
    """
    Generates a visualization of the computation graph using graphviz.
    Requires the `get_all_nodes_and_edges` function to be implemented correctly.

    Args:
        root_node (Value): The final node of the graph to visualize (e.g., loss).
        format (str): Output format ('svg', 'png', etc.). Default 'svg'.
        rankdir (str): Graph layout direction ('LR' or 'TB'). Default 'LR'.

    Returns:
        graphviz.Digraph: The graph object ready for rendering.
    """
    assert rankdir in ['LR', 'TB']
    # Call the student's implemented traversal function
    nodes, edges = get_all_nodes_and_edges(root_node)


    # Initialize graphviz object
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir})

    # Create nodes in the graphviz object
    for n in nodes:
        uid = str(id(n)) # Unique ID for the Value node

        # Format data and gradient strings based on shape
        data_str = f"shape={n.data.shape}" if hasattr(n, 'data') and isinstance(n.data, np.ndarray) and n.data.ndim > 0 else f"{getattr(n, 'data', '?'):.4f}"
        grad_str = f"shape={n.grad.shape}" if hasattr(n, 'grad') and isinstance(n.grad, np.ndarray) and n.grad.ndim > 0 else f"{getattr(n, 'grad', '?'):.4f}"
        label_str = f" | {getattr(n, 'label', '')}" if getattr(n, 'label', '') else ""
        # Create the label for the Value node rectangle
        node_label = f"{{ data {data_str} | grad {grad_str}{label_str} }}"
        # Add the Value node
        dot.node(name=uid, label=node_label, shape='record')

        # If this Value node was created by an operation, add an op node
        op = getattr(n, '_op', '')
        if op:
            op_uid = uid + op # Unique ID for the operation node
            dot.node(name=op_uid, label=op) # Add the operation node (oval)
            dot.edge(op_uid, uid) # Edge from Op -> Value

    # Create edges in the graphviz object
    for n1, n2 in edges: # Edge: parent (n1) -> child (n2)
        # Connect parent Value node to the operation node of the child
        parent_uid = str(id(n1))
        child_op = getattr(n2, '_op', '')
        if child_op: # Only draw edge if child has an associated operation
            child_op_uid = str(id(n2)) + child_op
            dot.edge(parent_uid, child_op_uid)

    return dot


#  Example Usage ---
# This block demonstrates how students can use the draw_dot function
# after implementing get_all_nodes_and_edges and their Value class ops.
if __name__ == '__main__':
    print("\n--- Visualization Example ---")
    # Simple expression: d = a*b + c*a
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    f = c*a; f.label='f'
    d = e + f; d.label='d'

    # Perform a backward pass to calculate gradients (optional for visualization)
    # try:
    #     d.backward()
    # except Exception as e:
    #     print(f"Note: Backward pass failed in example: {e}. Visualization might still work.")

    print("Generating example computation graph...")
    # Generate the graph visualization starting from the final node 'd'
    dot_graph = draw_dot(d)

    if dot_graph:
        # Render the graph to a file (e.g., 'example_graph.svg')
        # Requires Graphviz executables in system PATH
        try:
            output_filename = 'example_computation_graph'
            dot_graph.render(output_filename, view=False)
            print(f"Example graph saved as {output_filename}.* (e.g., .svg or .png)")
            print("Please include this generated graph in your report for Problem 4.")
        except graphviz.backend.execute.ExecutableNotFound:
            print("\n--- Graphviz Error ---")
            print("Graphviz executable not found. Visualization not saved.")
            print("Please install Graphviz (from www.graphviz.org)")
            print("and ensure the 'dot' command is available in your system's PATH.")
            print("----------------------\n")
        except Exception as e:
            print(f"An error occurred during graph rendering: {e}")
    else:
        print("Graph generation failed (likely due to traversal error).")
