# simple_trees.py - Animations simplifiées pour le tri
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

class Heap:
    def __init__(self, heap_type='max'):
        self.heap = []
        self.heap_type = heap_type
    
    def insert(self, key):
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, index):
        if index == 0:
            return
        parent = (index - 1) // 2
        if self.heap_type == 'max':
            if self.heap[index] > self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                self._heapify_up(parent)
    
    def heap_sort_animation(self):
        """Animation complète du tri par tas"""
        steps = []
        arr = self.heap.copy()
        n = len(arr)
        
        # Étape 1: Début
        steps.append({
            'action': 'start',
            'message': f'🚀 Début du tri par tas - {n} éléments: {arr}',
            'array': arr.copy(),
            'heap_size': n
        })
        
        # Étape 2: Construction du tas max
        steps.append({
            'action': 'build_heap',
            'message': '🏗️ Construction du tas max...',
            'array': arr.copy(),
            'heap_size': n
        })
        
        # Construire le tas max
        for i in range(n//2 - 1, -1, -1):
            self._heapify_animation(arr, n, i, steps)
        
        # Étape 3: Extraction des éléments
        steps.append({
            'action': 'start_extract',
            'message': '📤 Extraction des éléments un par un...',
            'array': arr.copy(),
            'heap_size': n
        })
        
        # Extraire les éléments un par un
        for i in range(n-1, 0, -1):
            # Échanger racine avec dernier élément
            steps.append({
                'action': 'swap_root',
                'message': f'🔄 Échange racine ({arr[0]}) avec position {i} ({arr[i]})',
                'array': arr.copy(),
                'swapped_indices': [0, i],
                'heap_size': i+1
            })
            arr[0], arr[i] = arr[i], arr[0]
            
            # Réorganiser le tas réduit
            steps.append({
                'action': 'heapify_down',
                'message': f'⚡ Réorganisation du tas (taille: {i})',
                'array': arr.copy(),
                'heap_size': i,
                'highlighted_index': 0
            })
            self._heapify_animation(arr, i, 0, steps)
        
        # Étape finale
        steps.append({
            'action': 'complete',
            'message': f'✅ Tri terminé! Tableau trié: {arr}',
            'array': arr.copy(),
            'heap_size': 0
        })
        
        return steps
    
    def _heapify_animation(self, arr, n, i, steps):
        """Animation de heapify down"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        # Étape de comparaison
        comparing = [i]
        if left < n:
            comparing.append(left)
        if right < n:
            comparing.append(right)
            
        steps.append({
            'action': 'compare',
            'message': f'🔍 Comparaison: nœud {i}({arr[i]}) avec enfants',
            'array': arr.copy(),
            'comparing_indices': comparing,
            'heap_size': n
        })
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
            
        if largest != i:
            # Étape d'échange
            steps.append({
                'action': 'swap',
                'message': f'🔄 Échange: {arr[i]} ↔ {arr[largest]}',
                'array': arr.copy(),
                'swapped_indices': [i, largest],
                'heap_size': n
            })
            arr[i], arr[largest] = arr[largest], arr[i]
            
            # Étape récursive
            steps.append({
                'action': 'recursive_heapify',
                'message': f'📊 Réorganisation à partir de nœud {largest}',
                'array': arr.copy(),
                'heap_size': n,
                'highlighted_index': largest
            })
            self._heapify_animation(arr, n, largest, steps)

class BTree:
    def __init__(self, order=3):
        self.order = order
        self.root = {'keys': [], 'children': [], 'leaf': True}
        self.node_counter = 0
    
    def insert(self, key):
        self._insert(self.root, key)
    
    def _insert(self, node, key):
        # Insertion simple pour la démo
        if len(node['keys']) < self.order - 1:
            node['keys'].append(key)
            node['keys'].sort()
        else:
            # Simulation de split
            node['keys'].append(key)
            node['keys'].sort()
            if not node['children']:
                node['children'] = [{'keys': [], 'children': [], 'leaf': True} for _ in range(2)]
    
    def inorder_traversal_animation(self):
        """Animation du parcours inorder du B-Arbre"""
        steps = []
        result = []
        
        def traverse(node, path="Racine", level=0):
            if not node['keys']:
                return
            
            # Étape: Entrée dans le nœud
            steps.append({
                'action': 'enter_node',
                'message': f'🚪 Entrée dans {path}: {node["keys"]}',
                'node_keys': node['keys'].copy(),
                'path': path,
                'level': level,
                'is_leaf': node['leaf']
            })
            
            if node['leaf']:
                # Feuille: traiter toutes les clés
                for i, key in enumerate(node['keys']):
                    result.append(key)
                    steps.append({
                        'action': 'visit_leaf_key',
                        'message': f'🍃 Visite clé feuille: {key}',
                        'current_key': key,
                        'sorted_result': result.copy(),
                        'path': path,
                        'key_index': i
                    })
            else:
                # Nœud interne: parcours récursif
                for i, key in enumerate(node['keys']):
                    if i < len(node['children']):
                        traverse(node['children'][i], f"{path}→Enfant{i}", level+1)
                    
                    result.append(key)
                    steps.append({
                        'action': 'visit_internal_key',
                        'message': f'📂 Visite clé interne: {key}',
                        'current_key': key,
                        'sorted_result': result.copy(),
                        'path': path,
                        'key_index': i
                    })
                
                # Dernier enfant
                if len(node['children']) > len(node['keys']):
                    traverse(node['children'][-1], f"{path}→DernierEnfant", level+1)
            
            # Étape: Sortie du nœud
            steps.append({
                'action': 'leave_node',
                'message': f'👋 Sortie de {path}',
                'node_keys': node['keys'].copy(),
                'path': path,
                'level': level
            })
        
        traverse(self.root)
        
        # Étape finale
        steps.append({
            'action': 'complete',
            'message': f'✅ Parcours terminé! Valeurs triées: {result}',
            'sorted_result': result.copy()
        })
        
        return steps

def draw_heap_animation(heap, step_data):
    """Dessine une étape de l'animation du tas"""
    plt.figure(figsize=(12, 8))
    
    arr = step_data.get('array', [])
    n = len(arr)
    heap_size = step_data.get('heap_size', n)
    
    # Graphique principal: Structure arborescente
    for i in range(heap_size):
        # Calcul position dans l'arbre
        level = 0
        temp = i + 1
        while temp > 1:
            level += 1
            temp //= 2
        
        nodes_in_level = 2 ** level
        pos_in_level = i - (2 ** level - 1)
        x = (pos_in_level + 0.5) * (10 / nodes_in_level)
        y = -level * 2
        
        # Couleur selon l'action
        if step_data['action'] in ['swap', 'swap_root'] and i in step_data.get('swapped_indices', []):
            color = 'red'
        elif step_data['action'] == 'compare' and i in step_data.get('comparing_indices', []):
            color = 'yellow'
        elif step_data['action'] == 'heapify_down' and i == step_data.get('highlighted_index', -1):
            color = 'orange'
        elif i >= heap_size:
            color = 'lightgray'  # Élements déjà triés
        else:
            color = 'lightblue'
        
        # Dessiner le nœud
        circle = plt.Circle((x, y), 0.4, color=color, ec='black', linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(x, y, str(arr[i]), ha='center', va='center', 
                fontweight='bold', fontsize=10)
        
        # Dessiner les arêtes
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < heap_size:
            level_left = (left + 1).bit_length() - 1
            pos_in_level_left = left - (2 ** level_left - 1)
            x_left = (pos_in_level_left + 0.5) * (10 / (2 ** level_left))
            y_left = -level_left * 2
            plt.plot([x, x_left], [y, y_left], 'k-', alpha=0.6, linewidth=1.5)
            
        if right < heap_size:
            level_right = (right + 1).bit_length() - 1
            pos_in_level_right = right - (2 ** level_right - 1)
            x_right = (pos_in_level_right + 0.5) * (10 / (2 ** level_right))
            y_right = -level_right * 2
            plt.plot([x, x_right], [y, y_right], 'k-', alpha=0.6, linewidth=1.5)
    
    plt.xlim(0, 10)
    plt.ylim(-10, 2)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    
    # Titre avec message
    plt.title(f"Tri par Tas - {step_data['message']}", 
              fontsize=14, fontweight='bold', pad=20)
    
    # Légende
    legend_text = f"Éléments: {arr}\nTaille tas: {heap_size}"
    plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def draw_btree_animation(btree, step_data):
    """Dessine une étape de l'animation du B-Arbre"""
    plt.figure(figsize=(10, 8))
    
    # Dessin simple pour B-Tree
    if step_data['action'] == 'enter_node':
        plt.text(0.5, 0.7, f"🚪 Entrée dans: {step_data['path']}", 
                fontsize=16, ha='center', fontweight='bold')
        plt.text(0.5, 0.5, f"Clés: {step_data['node_keys']}", 
                fontsize=14, ha='center', bbox=dict(boxstyle="round", facecolor="lightblue"))
    
    elif step_data['action'] in ['visit_leaf_key', 'visit_internal_key']:
        plt.text(0.5, 0.7, f"📂 Visite clé: {step_data['current_key']}", 
                fontsize=16, ha='center', fontweight='bold')
        plt.text(0.5, 0.5, f"Chemin: {step_data['path']}", 
                fontsize=12, ha='center')
        plt.text(0.5, 0.3, f"Résultat trié: {step_data['sorted_result']}", 
                fontsize=12, ha='center', bbox=dict(boxstyle="round", facecolor="lightgreen"))
    
    elif step_data['action'] == 'leave_node':
        plt.text(0.5, 0.5, f"👋 Sortie de: {step_data['path']}", 
                fontsize=16, ha='center', fontweight='bold')
        plt.text(0.5, 0.3, f"Clés: {step_data['node_keys']}", 
                fontsize=14, ha='center', bbox=dict(boxstyle="round", facecolor="lightyellow"))
    
    elif step_data['action'] == 'complete':
        plt.text(0.5, 0.5, f"✅ {step_data['message']}", 
                fontsize=16, ha='center', fontweight='bold', 
                bbox=dict(boxstyle="round", facecolor="lightgreen", pad=10))
    
    plt.text(0.5, 0.9, "Animation Parcours B-Arbre", 
            fontsize=18, ha='center', fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64