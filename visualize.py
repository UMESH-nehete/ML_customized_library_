# visualize.py

import graphviz
# Assuming Value is in my_ml_lib.nn.autograd
from my_ml_lib.nn.autograd import Value 
import numpy as np
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files (x86)\Graphviz\bin"



def get_all_nodes_and_edges(root_node: Value):
    """
    Performs a backward traversal from the root_node
    to find all unique Value nodes and the directed edges connecting them
    in the computation graph (Problem 4.4a).

    Args:
        root_node (Value): The final node in the graph (e.g., the loss Value object).

    Returns:
        tuple: (nodes, edges)
               nodes (set): A set containing all Value objects found during traversal.
               edges (set): A set of tuples (parent_Value, child_Value) representing
                            the directed edges: parent -> child.
    """
    # --- Step 1 - Initialize Sets ---
    nodes = set()
    edges = set()
    visited = set()


    # --- Step 2 - Implement DFS Traversal Function (`build_sets`) ---
    def build_sets(node):
        if node in visited:
            return
        
        visited.add(node)
        nodes.add(node)
        
        # _prev contains the parents of the current node (the operands)
        for parent in node._prev:
            # Add the directed edge: parent -> current node
            edges.add((parent, node)) 
            # Recursively call on the parent
            build_sets(parent)

    # --- Step 3 - Start Traversal ---
    build_sets(root_node) 

    # --- Step 4 - Return Results ---
    return nodes, edges
# --- End of Traversal Logic ---


# --- Graph Drawing Function (Keep the rest of the file as is) ---
def draw_dot(root_node: Value, format='png', rankdir='LR'):
    """
    Generates a visualization of the computation graph using graphviz.
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = get_all_nodes_and_edges(root_node)

    # Initialize graphviz object
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir})

    # Create nodes in the graphviz object
    for n in nodes:
        uid = str(id(n)) 

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
            op_uid = uid + op 
            dot.node(name=op_uid, label=op) 
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


# Example Usage ---
# The rest of the __main__ block remains the same for testing
if __name__ == '__main__':
    # ... [Example usage code] ...
    print("\n--- Visualization Example ---")
    # Simple expression: d = a*b + c*a
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    f = c*a; f.label='f'
    d = e + f; d.label='d'

    print("Generating example computation graph...")
    dot_graph = draw_dot(d)

    if dot_graph:
        try:
            output_filename = 'computation_graph'
            dot_graph.render(output_filename, view=False)
            print(f"Example graph saved as {output_filename}.* (e.g., .svg or .png)")
            print("Please include this generated graph in your report for Problem 4.")
        except Exception as e:
            print(f"An error occurred during graph rendering: {e}")
    else:
        print("Graph generation failed (likely due to traversal error).")