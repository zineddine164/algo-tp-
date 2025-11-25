from django.shortcuts import render
from django import forms
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, re
from . import simple_trees  

# ============================
# === FORMULAIRE ARBRE SIMPLE
# ============================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, re
import networkx as nx
from django import forms
from django.shortcuts import render


# ================================
# === FORMULAIRE POUR L’ABR =====
# ================================

class BSTForm(forms.Form):
    values = forms.CharField(
        label="Valeurs initiales de l’arbre binaire (ex: 8,3,10,1,6,14,4,7,13)",
        widget=forms.TextInput(attrs={'placeholder': 'Ex : 8,3,10,1,6,14,4,7,13'})
    )
    add_value = forms.IntegerField(
        label="Ajouter un nœud", required=False,
        widget=forms.NumberInput(attrs={'placeholder': 'Valeur à insérer'})
    )
    search_value = forms.IntegerField(
        label="Rechercher une valeur", required=False,
        widget=forms.NumberInput(attrs={'placeholder': 'Valeur à rechercher'})
    )
    delete_value = forms.IntegerField(
        label="Supprimer une valeur", required=False,
        widget=forms.NumberInput(attrs={'placeholder': 'Valeur à supprimer'})
    )


# ====================================
# === STRUCTURE : ARBRE BINAIRE ======
# ====================================

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


def insert(root, key):
    if root is None:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    return root


def delete(root, key):
    if root is None:
        return root
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # cas trouvé
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        min_node = minValueNode(root.right)
        root.key = min_node.key
        root.right = delete(root.right, min_node.key)
    return root


def minValueNode(node):
    current = node
    while current.left:
        current = current.left
    return current


def inorder(root):
    return inorder(root.left) + [root.key] + inorder(root.right) if root else []


def bst_to_edges(root):
    edges = []
    if root:
        if root.left:
            edges.append((root.key, root.left.key))
            edges += bst_to_edges(root.left)
        if root.right:
            edges.append((root.key, root.right.key))
            edges += bst_to_edges(root.right)
    return edges


def height(root):
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))


# ====================================
# === VUE PRINCIPALE DE L’ABR ========
# ====================================

def tree_view(request):
    form = BSTForm(request.POST or None)
    graph_img = None
    message = ""
    props = {}

    if request.method == "POST" and form.is_valid():
        values = [int(v.strip()) for v in re.split('[,; ]+', form.cleaned_data['values']) if v.strip().isdigit()]
        root = None
        for v in values:
            root = insert(root, v)

        action = request.POST.get("action")

        if action == "add":
            val = form.cleaned_data['add_value']
            if val is not None:
                if val in inorder(root):
                    message = f"⚠️ La valeur {val} existe déjà."
                else:
                    root = insert(root, val)
                    message = f"✅ Valeur {val} ajoutée à l’arbre."
            color_map = lambda n: "lightgreen" if n == val else "skyblue"

        elif action == "search":
            val = form.cleaned_data['search_value']
            found = val in inorder(root)
            message = f"✅ La valeur {val} existe dans l’arbre." if found else f"❌ La valeur {val} n’existe pas."
            color_map = lambda n: "lightgreen" if n == val else "skyblue"

        elif action == "delete":
            val = form.cleaned_data['delete_value']
            if val in inorder(root):
                root = delete(root, val)
                message = f"🗑️ Valeur {val} supprimée."
            else:
                message = f"⚠️ Valeur {val} introuvable."
            color_map = lambda n: "skyblue"

        else:
            color_map = lambda n: "skyblue"

        # === Dessin du graphe ===
        edges = bst_to_edges(root)
        G = nx.DiGraph()
        G.add_edges_from(edges)

        pos = hierarchy_pos(G, root.key)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True,
                node_color=[color_map(n) for n in G.nodes()],
                node_size=1500, font_size=10)
        plt.title("Arbre Binaire de Recherche (ABR)")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_img = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # === Propriétés ===
        if root:
            vals = inorder(root)
            props = {
                "Nombre de nœuds": len(vals),
                "Hauteur": height(root),
                "Minimum": min(vals),
                "Maximum": max(vals),
                "Parcours In-Order": ", ".join(map(str, vals))
            }

    return render(request, "matrix_structures/tree_view.html", {
        "form": form,
        "graph_img": graph_img,
        "props": props,
        "message": message
    })

# ======================================
# === POSITION HIÉRARCHIQUE (dessin) ===
# ======================================

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {}
    def _hierarchy_pos(G, node, left, right, vert_loc, pos):
        pos[node] = ((left + right) / 2, vert_loc)
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 2:
            _hierarchy_pos(G, neighbors[0], left, (left + right) / 2, vert_loc - vert_gap, pos)
            _hierarchy_pos(G, neighbors[1], (left + right) / 2, right, vert_loc - vert_gap, pos)
        elif len(neighbors) == 1:
            _hierarchy_pos(G, neighbors[0], left, right, vert_loc - vert_gap, pos)
        return pos
    return _hierarchy_pos(G, root, 0, width, vert_loc, pos)

# ====================================
# === FORMULAIRE POUR LES GRAPHES ====
# ====================================

class GraphChoiceForm(forms.Form):
    edges = forms.CharField(
        label="Arêtes du graphe (ex: A-B,B-C,C-D)",
        widget=forms.TextInput(attrs={'placeholder': 'Ex : A-B,B-C,C-D'})
    )
    GRAPH_TYPES = [
        ('non_oriente', 'Non orienté'),
        ('oriente', 'Orienté'),
    ]
    graph_type = forms.ChoiceField(
        label="Type de graphe",
        choices=GRAPH_TYPES,
        widget=forms.RadioSelect
    )

def graph_view(request):
    form = GraphChoiceForm(request.POST or None)
    graph_img = None
    props = {}

    if request.method == "POST" and form.is_valid():
        edges_input = form.cleaned_data['edges']
        graph_type = form.cleaned_data['graph_type']

        # Extraction et nettoyage des arêtes
        edges = [tuple(e.strip().split('-')) for e in re.split('[,;]', edges_input) if '-' in e]

        # Création du graphe
        if graph_type == 'oriente':
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from(edges)

        # === Propriétés du graphe ===
        if G.number_of_nodes() > 0:
            props = {
                "Type": graph_type,
                "Sommets": G.number_of_nodes(),
                "Arêtes": G.number_of_edges(),
                "Densité": round(nx.density(G), 3),
                "Degré moyen": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2)
            }

        # === Dessin du graphe ===
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6, 4))
        nx.draw(G, pos, with_labels=True, node_color="lightblue",
                node_size=1500, font_size=12)
        plt.title(f"Graphe : {graph_type}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graph_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return render(request, "matrix_structures/graph_view.html", {
        "form": form,
        "graph_img": graph_img,
        "props": props
    })


# =====================================
# === POSITION HIÉRARCHIQUE ===========
# =====================================
def hierarchy_pos(G, root, width=1., vert_gap=0.3, vert_loc=0,
                  xcenter=0.5, pos=None, parent=None):
    """Position hiérarchique pour un arbre non orienté."""
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    children = list(G.neighbors(root))
    if parent is not None and parent in children:
        children.remove(parent)

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                pos=pos, parent=root)
    return pos
from django.shortcuts import render


def principal(request):
    """Page principale contenant les liens vers TP1 → TP6"""
    return render(request, "principal/principal.html")


def tp1(request):
    return render(request, "matrix_structures/tp1.html")


def tp2(request):
    return render(request, "matrix_structures/tp2.html")


def tp3(request):
    return render(request, "matrix_structures/tp3.html")


def tp4(request):
    return render(request, "matrix_structures/tp4.html")


def tp5(request):
    return render(request, "matrix_structures/tp5.html")
def tp6(request):
    return render(request, "matrix_structures/tp6.html")

def affiche_timsort(request):
    return render(request, "matrix_structures/affiche_timsort.html")



# Dans views.py
from . import tree_implementations

def visualisation(request):
    context = {}
    form = tree_implementations.TreeVisualizationForm(request.POST or None, request.FILES or None)
    
    # Gestion de la réinitialisation
    if request.method == "POST" and request.POST.get('action') == 'reset':
        if 'trees_data' in request.session:
            del request.session['trees_data']
        if 'animation_data' in request.session:
            del request.session['animation_data']
        form = tree_implementations.TreeVisualizationForm()
        context['form'] = form
        context['message'] = "🆕 Session réinitialisée"
        context['message_type'] = 'info'
        return render(request, "matrix_structures/visualisation.html", context)
    
    # Initialiser la session
    if 'trees_data' not in request.session:
        request.session['trees_data'] = {
            'values': [],
            'b_tree_order': 3
        }
    
    # Initialiser les données d'animation
    if 'animation_data' not in request.session:
        request.session['animation_data'] = {
            'current_step': 0,
            'total_steps': 0,
            'steps': [],
            'current_tree_type': None,
            'animation_type': None,
            'operation_value': None
        }
    
    session_data = request.session['trees_data']
    animation_data = request.session['animation_data']
    current_values = session_data.get('values', [])
    b_tree_order = session_data.get('b_tree_order', 3)
    
    # Gestion des animations
    if request.method == "POST" and request.POST.get('action') == 'animation_step':
        current_step = animation_data.get('current_step', 0)
        total_steps = animation_data.get('total_steps', 0)
        steps_dict = animation_data.get('steps', [])
        tree_type = animation_data.get('current_tree_type')
        anim_type = animation_data.get('animation_type')
        
        if steps_dict and current_step < total_steps:
            # Convertir le dictionnaire en objet AnimationStep
            step_dict = steps_dict[current_step]
            step = tree_implementations.AnimationStep.from_dict(step_dict)
            
            # Générer l'arbre pour l'animation
            if current_values:
                if tree_type == 'abr':
                    tree = tree_implementations.generate_abr(current_values)
                elif tree_type == 'avl':
                    tree = tree_implementations.generate_avl(current_values)
                elif tree_type == 'heap':
                    tree = tree_implementations.generate_heap(current_values)
                elif tree_type == 'btree':
                    tree = tree_implementations.generate_btree(current_values, b_tree_order)
                else:
                    tree = None
                
                if tree:
                    # Générer l'image animée
                    context['animation_img'] = tree_implementations.TreeAnimator.draw_animated_tree(
                        tree, step.to_dict(), f"Arbre {tree_type.upper()}"
                    )
                    context['animation_step'] = f"Étape {current_step + 1}/{total_steps}"
                    context['animation_message'] = step.message
                    context['current_tree_type'] = tree_type
                    context['animation_type'] = anim_type
                    
                    # Mettre à jour pour la prochaine étape
                    animation_data['current_step'] = current_step + 1
                    request.session.modified = True
                    
                    if current_step + 1 >= total_steps:
                        context['message'] = f"🎉 Animation {anim_type} terminée pour {tree_type.upper()}!"
                        context['message_type'] = 'success'
                    else:
                        context['message'] = f"⏭️ Étape suivante de {anim_type}..."
                        context['message_type'] = 'info'

    if request.method == "POST" and form.is_valid():
        # Mettre à jour l'ordre du B-Arbre si modifié
        new_b_tree_order = form.cleaned_data.get('b_tree_order')
        if new_b_tree_order and new_b_tree_order != b_tree_order:
            b_tree_order = new_b_tree_order
            session_data['b_tree_order'] = b_tree_order
            request.session.modified = True
        
        # Gestion fichier et saisie manuelle
        values = []
        new_values_loaded = False
        
        file_input = request.FILES.get('file_input')
        if file_input:
            content = file_input.read().decode('utf-8')
            values = [int(x) for x in re.findall(r'\d+', content)]
            new_values_loaded = True
        
        manual_input = form.cleaned_data.get('manual_input', '')
        if manual_input and not values:
            values = [int(x) for x in re.findall(r'\d+', manual_input)]
            new_values_loaded = True
        
        # Mettre à jour les valeurs si nouveau chargement
        if new_values_loaded and values:
            current_values = values
            session_data['values'] = current_values
            request.session.modified = True
        
        # Récupérer l'action et la valeur
        action = request.POST.get('action', 'generate')
        operation_value = form.cleaned_data.get('operation_value')
        
        try:
            # GÉNÉRER LES ARBRES AVEC LES VALEURS COURANTES
            if current_values:
                abr_tree = tree_implementations.generate_abr(current_values)
                avl_tree = tree_implementations.generate_avl(current_values)
                heap_tree = tree_implementations.generate_heap(current_values)
                btree_tree = tree_implementations.generate_btree(current_values, b_tree_order)
            else:
                abr_tree, avl_tree, heap_tree, btree_tree = None, None, None, None
            
            # GESTION DES ACTIONS
            if operation_value is not None:
                if action == "insert":
                    # INSERTION NORMALE
                    if operation_value not in current_values:
                        current_values.append(operation_value)
                        session_data['values'] = current_values
                        request.session.modified = True
                        
                        # Régénérer les arbres avec la nouvelle valeur
                        abr_tree = tree_implementations.generate_abr(current_values)
                        avl_tree = tree_implementations.generate_avl(current_values)
                        heap_tree = tree_implementations.generate_heap(current_values)
                        btree_tree = tree_implementations.generate_btree(current_values, b_tree_order)
                        
                        context['message'] = f"✅ Valeur {operation_value} insérée"
                        context['message_type'] = 'success'
                    else:
                        context['message'] = f"⚠️ Valeur {operation_value} existe déjà"
                        context['message_type'] = 'error'
                elif action == "search":
                    # RECHERCHE NORMALE
                    results = []
                    if abr_tree and abr_tree.search(operation_value):
                        results.append("ABR")
                    if avl_tree and avl_tree.search(operation_value):
                        results.append("AVL")
                    if heap_tree and heap_tree.search(operation_value):
                        results.append("Tas")
                    if btree_tree and btree_tree.search(operation_value):
                        results.append("B-Arbre")
                    
                    if results:
                        context['message'] = f"✅ Valeur {operation_value} trouvée dans: {', '.join(results)}"
                        context['message_type'] = 'success'
                    else:
                        context['message'] = f"❌ Valeur {operation_value} non trouvée"
                        context['message_type'] = 'error'
                
                elif action == "delete":
                    # SUPPRESSION NORMALE
                    if operation_value in current_values:
                        current_values = [v for v in current_values if v != operation_value]
                        session_data['values'] = current_values
                        request.session.modified = True
                        
                        # Régénérer les arbres sans la valeur supprimée
                        if current_values:
                            abr_tree = tree_implementations.generate_abr(current_values)
                            avl_tree = tree_implementations.generate_avl(current_values)
                            heap_tree = tree_implementations.generate_heap(current_values)
                            btree_tree = tree_implementations.generate_btree(current_values, b_tree_order)
                        else:
                            abr_tree, avl_tree, heap_tree, btree_tree = None, None, None, None
                        
                        context['message'] = f"🗑️ Valeur {operation_value} supprimée"
                        context['message_type'] = 'success'
                    else:
                        context['message'] = f"⚠️ Valeur {operation_value} introuvable"
                        context['message_type'] = 'error'
                
                elif action == "sort":
                    # TRI NORMAL
                    sorted_info = []
                    if heap_tree:
                        heap_sorted = heap_tree.heap_sort()
                        sorted_info.append(f"<strong>Tas (Heap Sort):</strong> {heap_sorted}")
                    if avl_tree:
                        avl_sorted = avl_tree.sort()
                        sorted_info.append(f"<strong>AVL (Inorder):</strong> {avl_sorted}")
                    if btree_tree:
                        btree_sorted = btree_tree.inorder()
                        sorted_info.append(f"<strong>B-Arbre (Inorder):</strong> {btree_sorted}")
                    
                    if sorted_info:
                        context['message'] = "📊 <strong>Résultats du tri:</strong><br>" + "<br>".join(sorted_info)
                        context['message_type'] = 'info'
                    else:
                        context['message'] = "❌ Aucun arbre disponible pour le tri"
                        context['message_type'] = 'error'
            
            # GÉNÉRER LES IMAGES ET PROPRIÉTÉS (TOUJOURS APRÈS LES ACTIONS)
            if abr_tree:
                context['abr_img'] = tree_implementations.draw_tree(abr_tree.root, 'ABR')
                context['abr_props'] = tree_implementations.abr_properties(abr_tree)
            
            if avl_tree:
                context['avl_img'] = tree_implementations.draw_tree(avl_tree.root, 'AVL')
                context['avl_props'] = tree_implementations.avl_properties(avl_tree)
            
            if heap_tree:
                heap_root = heap_tree.to_tree_structure()
                context['heap_img'] = tree_implementations.draw_tree(heap_root, 'Tas')
                context['heap_props'] = tree_implementations.heap_properties(heap_tree)
            
            if btree_tree:
                context['btree_img'] = tree_implementations.draw_btree(btree_tree)
                context['btree_props'] = tree_implementations.btree_properties(btree_tree)
            
            # COMPARAISON DES PERFORMANCES
            context['comparison'] = {
                # Hauteurs
                'abr_height': context.get('abr_props', {}).get('Hauteur', 0),
                'avl_height': context.get('avl_props', {}).get('Hauteur', 0), 
                'heap_height': context.get('heap_props', {}).get('Hauteur', 0),
                'btree_height': context.get('btree_props', {}).get('Hauteur', 0),
                
                # Profondeurs moyennes
                'abr_avg_depth': context.get('abr_props', {}).get('Profondeur moyenne', 0),
                'avl_avg_depth': context.get('avl_props', {}).get('Profondeur moyenne', 0),
                'heap_avg_depth': context.get('heap_props', {}).get('Profondeur moyenne', 0),
                'btree_avg_depth': context.get('btree_props', {}).get('Profondeur moyenne', 0),
                
                # Qualité d'équilibre
                'abr_balance_quality': context.get('abr_props', {}).get('Qualité équilibre', '-'),
                'avl_balance_quality': context.get('avl_props', {}).get('Qualité équilibre', '-'),
                'heap_balance_quality': context.get('heap_props', {}).get('Qualité équilibre', '-'),
                'btree_balance_quality': context.get('btree_props', {}).get('Qualité équilibre', '-'),
            }
            
            # MESSAGE PAR DÉFAUT SI AUCUN MESSAGE N'EST DÉJÀ SET
            if not context.get('message'):
                if current_values:
                    context['message'] = f"✅ {len(current_values)} valeurs dans les arbres"
                    context['message_type'] = 'success'
                else:
                    context['message'] = "❌ Aucune valeur chargée"
                    context['message_type'] = 'error'
                
        except Exception as e:
            context['message'] = f"❌ Erreur: {str(e)}"
            context['message_type'] = 'error'
    
    else:
        # POUR LES REQUÊTES GET - CHARGER LES DONNÉES EXISTANTES
        if current_values:
            abr_tree = tree_implementations.generate_abr(current_values)
            avl_tree = tree_implementations.generate_avl(current_values)
            heap_tree = tree_implementations.generate_heap(current_values)
            btree_tree = tree_implementations.generate_btree(current_values, b_tree_order)
            
            # Génération des images et propriétés
            if abr_tree:
                context['abr_img'] = tree_implementations.draw_tree(abr_tree.root, 'ABR')
                context['abr_props'] = tree_implementations.abr_properties(abr_tree)
            
            if avl_tree:
                context['avl_img'] = tree_implementations.draw_tree(avl_tree.root, 'AVL')
                context['avl_props'] = tree_implementations.avl_properties(avl_tree)
            
            if heap_tree:
                heap_root = heap_tree.to_tree_structure()
                context['heap_img'] = tree_implementations.draw_tree(heap_root, 'Tas')
                context['heap_props'] = tree_implementations.heap_properties(heap_tree)
            
            if btree_tree:
                context['btree_img'] = tree_implementations.draw_btree(btree_tree)
                context['btree_props'] = tree_implementations.btree_properties(btree_tree)
            
            # Comparaison
            context['comparison'] = {
                'abr_height': context.get('abr_props', {}).get('Hauteur', 0),
                'avl_height': context.get('avl_props', {}).get('Hauteur', 0), 
                'heap_height': context.get('heap_props', {}).get('Hauteur', 0),
                'btree_height': context.get('btree_props', {}).get('Hauteur', 0),
                'abr_avg_depth': context.get('abr_props', {}).get('Profondeur moyenne', 0),
                'avl_avg_depth': context.get('avl_props', {}).get('Profondeur moyenne', 0),
                'heap_avg_depth': context.get('heap_props', {}).get('Profondeur moyenne', 0),
                'btree_avg_depth': context.get('btree_props', {}).get('Profondeur moyenne', 0),
                'abr_balance_quality': context.get('abr_props', {}).get('Qualité équilibre', '-'),
                'avl_balance_quality': context.get('avl_props', {}).get('Qualité équilibre', '-'),
                'heap_balance_quality': context.get('heap_props', {}).get('Qualité équilibre', '-'),
                'btree_balance_quality': context.get('btree_props', {}).get('Qualité équilibre', '-'),
            }
    
    # INITIALISATION FINALE DU FORMULAIRE
    form.fields['b_tree_order'].initial = b_tree_order
    context['form'] = form
    context['initial_values'] = ', '.join(map(str, current_values)) if current_values else ''
    
    return render(request, "matrix_structures/visualisation.html", context)
def simple_visualisation(request):
    """Version simplifiée avec seulement les animations de tri"""
    context = {}
    
    # Données de session
    if 'simple_tree_data' not in request.session:
        request.session['simple_tree_data'] = {'values': []}
    
    current_values = request.session['simple_tree_data']['values']
    
    if request.method == "POST":
        # Charger les valeurs
        if 'load_values' in request.POST:
            values_text = request.POST.get('values_input', '')
            if values_text:
                import re
                current_values = [int(x) for x in re.findall(r'\d+', values_text)]
                request.session['simple_tree_data']['values'] = current_values
                request.session.modified = True
                context['message'] = f"✅ {len(current_values)} valeurs chargées"
        
        # Animation TRI PAR TAS
        elif 'heap_animation' in request.POST and current_values:
            from . import simple_trees
            
            heap = simple_trees.Heap()
            for val in current_values:
                heap.insert(val)
            
            steps = heap.heap_sort_animation()
            request.session['simple_animation'] = {
                'steps': steps,
                'current_step': 0,
                'type': 'heap_sort',
                'heap': heap.heap.copy()  # Sauvegarder le tas
            }
            request.session.modified = True
            
            # Afficher première étape
            step = steps[0]
            context['animation_img'] = simple_trees.draw_heap_animation(heap, step)
            context['animation_message'] = step['message']
            context['show_animation'] = True
            context['current_values'] = ', '.join(map(str, current_values))
        
        # Animation B-ARBRE
        elif 'btree_animation' in request.POST and current_values:
            from . import simple_trees
            
            btree = simple_trees.BTree()
            for val in current_values:
                btree.insert(val)
            
            steps = btree.inorder_traversal_animation()
            request.session['simple_animation'] = {
                'steps': steps,
                'current_step': 0, 
                'type': 'btree_traversal'
            }
            request.session.modified = True
            
            # Afficher première étape
            step = steps[0]
            context['animation_img'] = simple_trees.draw_btree_animation(btree, step)
            context['animation_message'] = step['message']
            context['show_animation'] = True
            context['current_values'] = ', '.join(map(str, current_values))
        
        # Étape suivante
        elif 'next_step' in request.POST and 'simple_animation' in request.session:
            animation_data = request.session['simple_animation']
            current_step = animation_data['current_step']
            steps = animation_data['steps']
            
            if current_step < len(steps) - 1:
                current_step += 1
                step = steps[current_step]
                
                # Régénérer l'arbre pour l'animation
                if animation_data['type'] == 'heap_sort':
                    from . import simple_trees
                    heap = simple_trees.Heap()
                    heap.heap = animation_data.get('heap', [])
                    context['animation_img'] = simple_trees.draw_heap_animation(heap, step)
                else:  # btree_traversal
                    from . import simple_trees
                    btree = simple_trees.BTree()
                    for val in current_values:
                        btree.insert(val)
                    context['animation_img'] = simple_trees.draw_btree_animation(btree, step)
                
                context['animation_message'] = step['message']
                context['show_animation'] = True
                context['current_values'] = ', '.join(map(str, current_values))
                
                # Mettre à jour la session
                animation_data['current_step'] = current_step
                request.session.modified = True
                
                if current_step >= len(steps) - 1:
                    context['message'] = "🎉 Animation terminée!"
    
    context['current_values'] = ', '.join(map(str, current_values)) if current_values else ''
    
    return render(request, "matrix_structures/simple_visualisation.html", context)
from django.shortcuts import render

# ===================================================================
#   TIMSORT : EXTRACTION DES ETAPES PRINCIPALES POUR L'ARBRE
# ===================================================================
from django.shortcuts import render

from django.shortcuts import render

# ===================== TIMSORT =====================
def timsort_main_steps(arr, minrun):
    steps = []
    n = len(arr)

    def snapshot(label, run_left=None, run_right=None, complexity="O(1)"):
        run_block = None
        inserted = []
        if run_left is not None and run_right is not None:
            run_block = arr[run_left:run_right+1]
            inserted = run_block.copy()
        steps.append({
            "label": label,
            "array": arr.copy(),
            "complexity": complexity,
            "run_left": run_left,
            "run_right": run_right,
            "run_block": run_block,
            "inserted": inserted
        })

    # 1) Découpage + insertion sort
    i = 0
    while i < n:
        left = i
        right = min(i + minrun - 1, n - 1)
        snapshot(f"📌 Détection du run [{left}:{right}]", left, right, f"O({right-left+1}²)")

        # Tri par insertion sur le run
        for j in range(left + 1, right + 1):
            key = arr[j]
            k = j - 1
            while k >= left and arr[k] > key:
                arr[k + 1] = arr[k]
                k -= 1
            arr[k + 1] = key

        snapshot(f"🔵 Run [{left}:{right}] trié", left, right, f"O({right-left+1}²)")
        i += minrun

    snapshot("🟦 Tous les runs initiaux sont triés", complexity="O(n)")

    # 2) Fusion
    size = minrun
    while size < n:
        for left in range(0, n, 2*size):
            mid = min(left + size - 1, n - 1)
            right = min(left + 2*size - 1, n - 1)

            if mid < right:
                # Fusion réelle
                merged = []
                l, r = left, mid + 1
                while l <= mid and r <= right:
                    if arr[l] <= arr[r]:
                        merged.append(arr[l]); l += 1
                    else:
                        merged.append(arr[r]); r += 1
                while l <= mid:
                    merged.append(arr[l]); l += 1
                while r <= right:
                    merged.append(arr[r]); r += 1

                arr[left:left+len(merged)] = merged

                # Snapshot fusion et résultat après fusion
                snapshot(f"🟧 Fusion [{left}:{mid}] + [{mid+1}:{right}]", left, right, f"O({right-left+1})")
                snapshot(f"🟩 Résultat après fusion [{left}:{right}]", left, right, f"O({right-left+1})")

        size *= 2

    
    return steps

# ===================== CONSTRUCTION ARBRE 5 ÉTAPES =====================
def build_tree_levels_5steps(steps):
    """
    Regroupe les étapes en 5 niveaux :
    1. Détection des runs
    2. Runs triés
    3. Fusions
    4. Résultat des fusions
    5. Tableau final trié
    """
    levels = []

    detect = [s for s in steps if "Détection du run" in s["label"]]
    sorted_runs = [s for s in steps if "Run [" in s["label"] and "trié" in s["label"]]
    fusion = [s for s in steps if "Fusion [" in s["label"]]
    fusion_result = [s for s in steps if "Résultat après fusion" in s["label"]]
    final = [s for s in steps if "Tableau final trié" in s["label"]]

    if detect: levels.append(detect)
    if sorted_runs: levels.append(sorted_runs)
    if fusion: levels.append(fusion)
    if fusion_result: levels.append(fusion_result)
    if final: levels.append(final)

    return levels

# ===================== VUE DJANGO =====================
def affiche_timsort(request):
    steps = []
    input_list = ""
    minrun = 32

    if request.method == "POST":
        input_list = request.POST.get("liste", "")
        minrun_input = request.POST.get("minrun", "")
        try:
            minrun = int(minrun_input) if minrun_input else 32
        except:
            minrun = 32
        try:
            arr = [int(x.strip()) for x in input_list.split(",") if x.strip()]
            if not arr:
                raise ValueError
            steps = timsort_main_steps(arr, minrun)
        except ValueError:
            steps = [{
                "label": "❌ Erreur : entrez une liste correcte",
                "array": [],
                "run_block": None,
                "complexity": "",
                "inserted": []
            }]

    tree_levels = build_tree_levels_5steps(steps)

    return render(request, "matrix_structures/affiche_timsort.html", {
        "steps": steps,
        "tree_levels": tree_levels,
        "input_list": input_list,
        "minrun": minrun
    })
