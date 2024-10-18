import graphviz
import matplotlib.pyplot as plt

from .pedigrees import get_all_id_set


def render_ped(
    up_dct: dict[int, dict[int, int]],
    name: str,
    out_dir: str,
    color_dict=None,
    label_dict=None,
    focal_id=None,
):
    dot = graphviz.Digraph(name)
    all_id_set = get_all_id_set(up_dct)
    if color_dict is None:
        color_dict = {i : 'dodgerblue' for i in all_id_set if i > 0}
    if label_dict is None:
        label_dict = {n : str(n) for n in all_id_set}
    if focal_id is not None:
        color_dict[focal_id] = 'red'
    for n in all_id_set:
        edge_color = None
        fill_color = color_dict[n] if n in color_dict else None
        style = 'filled' if n in color_dict else None
        label = label_dict.get(n, "")
        dot.node(
            str(n),
            color=edge_color,
            fillcolor=fill_color,
            style=style,
            label=label,
        )
    for c,pset in up_dct.items():
        for p in pset:
            dot.edge(str(p), str(c), arrowhead='none')
    plt.clf()
    dot.render(directory=out_dir).replace('\\', '/')
