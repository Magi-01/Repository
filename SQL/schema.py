import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_erd(title, entities, relationships):
    fig, ax = plt.subplots(figsize=(12, 8))

    for entity in entities:
        rect = mpatches.FancyBboxPatch(entity['pos'], entity['width'], entity['height'],
                                        boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey")
        ax.add_patch(rect)
        ax.text(entity['pos'][0] + entity['width'] / 2, entity['pos'][1] + entity['height'] / 2,
                entity['name'], horizontalalignment='center', verticalalignment='center', fontsize=10, fontweight='bold')

    for relationship in relationships:
        ax.annotate("", xy=relationship['to'], xytext=relationship['from'],
                    arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.5),
                    annotation_clip=False)
        ax.text((relationship['from'][0] + relationship['to'][0]) / 2,
                (relationship['from'][1] + relationship['to'][1]) / 2,
                relationship['name'], horizontalalignment='center', verticalalignment='center', fontsize=8, fontstyle='italic')

    plt.xlim(-1, 13)
    plt.ylim(-1, 9)
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.show()

# Conceptual Schema
entities_conceptual = [
    {'name': 'Players', 'pos': [1, 6], 'width': 2, 'height': 1},
    {'name': 'Dimensions', 'pos': [5, 6], 'width': 2, 'height': 1},
    {'name': 'Items', 'pos': [9, 6], 'width': 2, 'height': 1},
    {'name': 'Travel', 'pos': [3, 3], 'width': 2, 'height': 1},
    {'name': 'NPCs', 'pos': [7, 3], 'width': 2, 'height': 1},
    {'name': 'PlayerItems', 'pos': [1, 1], 'width': 2, 'height': 1}
]

relationships_conceptual = [
    {'name': '1 to n', 'from': [1, 6.5], 'to': [3, 3.5]},
    {'name': '1 to n', 'from': [1, 6.5], 'to': [1, 1.5]},
    {'name': '1 to n', 'from': [5, 6.5], 'to': [9, 6.5]},
    {'name': '1 to n', 'from': [5, 6.5], 'to': [7, 3.5]},
    {'name': 'n to 1', 'from': [3, 3.5], 'to': [5, 6.5]},
    {'name': 'n to 1', 'from': [3, 3.5], 'to': [5, 6.5]},
    {'name': 'n to 1', 'from': [1, 1.5], 'to': [1, 6.5]},
    {'name': 'n to 1', 'from': [1, 1.5], 'to': [9, 6.5]}
]

draw_erd("Conceptual Schema", entities_conceptual, relationships_conceptual)
