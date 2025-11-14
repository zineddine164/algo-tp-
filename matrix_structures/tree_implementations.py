import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from django import forms
import time
from django.shortcuts import render
# ==========================================
# === FORMULAIRE POUR LA VISUALISATION ===
# ==========================================

class TreeVisualizationForm(forms.Form):
    file_input = forms.FileField(
        label="Charger depuis un fichier",
        required=False,
        widget=forms.FileInput(attrs={'accept': '.txt,.csv'})
    )
    manual_input = forms.CharField(
        label="Saisie manuelle",
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Ex: 50,30,70,20,40,60,80',
            'class': 'form-control'
        })
    )
    operation_value = forms.IntegerField(
        label="Valeur pour l'opération",
        required=False,
        widget=forms.NumberInput(attrs={
            'placeholder': 'Entrez une valeur numérique',
            'class': 'form-control'
        })
    )
    b_tree_order = forms.IntegerField(
        label="Ordre du B-Arbre",
        initial=3,
        min_value=2,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

# ==========================================
# === CLASSE DE BASE POUR LES ARBRES ======
# ==========================================

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # Pour AVL

class BTreeNode:
    def __init__(self, order, leaf=False):
        self.order = order
        self.keys = []
        self.children = []
        self.leaf = leaf

# ==========================================
# === ARBRE BINAIRE DE RECHERCHE (ABR) ====
# ==========================================

class ABR:
    def __init__(self):
        self.root = None
    
    def insert(self, key):
        self.root = self._insert(self.root, key)
    
    def _insert(self, root, key):
        if root is None:
            return Node(key)
        
        if key < root.key:
            root.left = self._insert(root.left, key)
        elif key > root.key:
            root.right = self._insert(root.right, key)
        
        return root
    
    def search(self, key):
        return self._search(self.root, key)
    
    def _search(self, root, key):
        if root is None or root.key == key:
            return root
        
        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)
    
    def delete(self, key):
      self.root = self._delete(self.root, key)

    def _delete(self, root, key):
      if root is None:
        return root
    
      if key < root.key:
        root.left = self._delete(root.left, key)
      elif key > root.key:
        root.right = self._delete(root.right, key)
      else:
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        
        temp = self._min_value_node(root.right)
        root.key = temp.key
        root.right = self._delete(root.right, temp.key)
    
      return root

    def _min_value_node(self, node):
       current = node
       while current.left:
        current = current.left
       return current
    
    def height(self):
        return self._height(self.root)
    
    def _height(self, node):
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
    
    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)

# ==========================================
# === ARBRE AVL ===========================
# ==========================================

class AVL:
    def __init__(self):
        self.root = None
    
    def insert(self, key):
        self.root = self._insert(self.root, key)
    
    def _insert(self, root, key):
        # Insertion standard ABR
        if not root:
            return Node(key)
        elif key < root.key:
            root.left = self._insert(root.left, key)
        else:
            root.right = self._insert(root.right, key)
        
        # Mise à jour de la hauteur
        root.height = 1 + max(self._get_height(root.left), 
                             self._get_height(root.right))
        
        # Équilibrage
        balance = self._get_balance(root)
        
        # Cas de déséquilibre gauche-gauche
        if balance > 1 and key < root.left.key:
            return self._rotate_right(root)
        
        # Cas de déséquilibre droite-droite
        if balance < -1 and key > root.right.key:
            return self._rotate_left(root)
        
        # Cas de déséquilibre gauche-droite
        if balance > 1 and key > root.left.key:
            root.left = self._rotate_left(root.left)
            return self._rotate_right(root)
        
        # Cas de déséquilibre droite-gauche
        if balance < -1 and key < root.right.key:
            root.right = self._rotate_right(root.right)
            return self._rotate_left(root)
        
        return root
    
    def _get_height(self, root):
        if not root:
            return 0
        return root.height
    
    def _get_balance(self, root):
        if not root:
            return 0
        return self._get_height(root.left) - self._get_height(root.right)
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        
        # Rotation
        y.left = z
        z.right = T2
        
        # Mise à jour des hauteurs
        z.height = 1 + max(self._get_height(z.left), 
                          self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), 
                          self._get_height(y.right))
        
        return y
    
    def _rotate_right(self, z):
        y = z.left
        T3 = y.right
        
        # Rotation
        y.right = z
        z.left = T3
        
        # Mise à jour des hauteurs
        z.height = 1 + max(self._get_height(z.left), 
                          self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), 
                          self._get_height(y.right))
        
        return y
    
    def search(self, key):
        return self._search(self.root, key)
    
    def _search(self, root, key):
        if root is None or root.key == key:
            return root
        
        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)
    
    def delete(self, key):
        self.root = self._delete(self.root, key)
    
    def _delete(self, root, key):
        # Suppression standard ABR
        if not root:
            return root
        
        if key < root.key:
            root.left = self._delete(root.left, key)
        elif key > root.key:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            
            temp = self._min_value_node(root.right)
            root.key = temp.key
            root.right = self._delete(root.right, temp.key)
        
        if root is None:
            return root
        
        # Mise à jour de la hauteur
        root.height = 1 + max(self._get_height(root.left),
                             self._get_height(root.right))
        
        # Équilibrage
        balance = self._get_balance(root)
        
        # Cas de déséquilibre
        if balance > 1 and self._get_balance(root.left) >= 0:
            return self._rotate_right(root)
        
        if balance < -1 and self._get_balance(root.right) <= 0:
            return self._rotate_left(root)
        
        if balance > 1 and self._get_balance(root.left) < 0:
            root.left = self._rotate_left(root.left)
            return self._rotate_right(root)
        
        if balance < -1 and self._get_balance(root.right) > 0:
            root.right = self._rotate_right(root.right)
            return self._rotate_left(root)
        
        return root
    
    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current
    
    def height(self):
        return self._get_height(self.root)
    
    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)

    def sort(self):
        return self.inorder()  # Déjà trié car ABR 

      

# ==========================================
# === TAS (HEAP) ==========================
# ==========================================

class Heap:
    def __init__(self, heap_type='max'):
        self.heap = []
        self.heap_type = heap_type  # 'max' ou 'min'
    
    def insert(self, key):
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, index):
        parent = (index - 1) // 2
        
        if self.heap_type == 'max':
            if index > 0 and self.heap[index] > self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                self._heapify_up(parent)
        else:
            if index > 0 and self.heap[index] < self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                self._heapify_up(parent)
    
    def delete(self, key):
        if key not in self.heap:
            return False
        
        index = self.heap.index(key)
        last = len(self.heap) - 1
        self.heap[index] = self.heap[last]
        self.heap.pop()
        
        if index < len(self.heap):
            self._heapify_down(index)
            self._heapify_up(index)
        
        return True
    
    def _heapify_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        extreme = index
        
        if self.heap_type == 'max':
            if left < len(self.heap) and self.heap[left] > self.heap[extreme]:
                extreme = left
            if right < len(self.heap) and self.heap[right] > self.heap[extreme]:
                extreme = right
        else:
            if left < len(self.heap) and self.heap[left] < self.heap[extreme]:
                extreme = left
            if right < len(self.heap) and self.heap[right] < self.heap[extreme]:
                extreme = right
        
        if extreme != index:
            self.heap[index], self.heap[extreme] = self.heap[extreme], self.heap[index]
            self._heapify_down(extreme)
    
    def search(self, key):
        return key in self.heap
    
    def height(self):
        n = len(self.heap)
        if n == 0:
            return 0
        height = 0
        while (1 << height) <= n:
            height += 1
        return height
    
    def to_tree_structure(self):
        """Convertit le tas en structure arborescente pour la visualisation"""
        if not self.heap:
            return None
        
        nodes = [Node(val) for val in self.heap]
        
        for i in range(len(nodes)):
            left_index = 2 * i + 1
            right_index = 2 * i + 2
            
            if left_index < len(nodes):
                nodes[i].left = nodes[left_index]
            if right_index < len(nodes):
                nodes[i].right = nodes[right_index]
        
        return nodes[0]  # Retourne la racine
    def heap_sort(self):
     if not self.heap:
        return []
     sorted_list = []
     temp_heap = self.heap.copy()
    
    # Pour un tas max, on extrait le maximum à chaque fois
     if self.heap_type == 'max':
        while temp_heap:
            # Extraire le maximum
            max_val = temp_heap[0]
            sorted_list.append(max_val)
            
            # Remplacer la racine par le dernier élément
            if len(temp_heap) > 1:
                temp_heap[0] = temp_heap.pop()
                # Réorganiser le tas
                self._heapify_down_custom(temp_heap, 0)
            else:
                temp_heap.pop()
        
        # Inverser pour avoir l'ordre croissant
        return sorted_list[::-1]
    
     else:  # Tas min
        while temp_heap:
            # Extraire le minimum
            min_val = temp_heap[0]
            sorted_list.append(min_val)
            
            # Remplacer la racine par le dernier élément
            if len(temp_heap) > 1:
                temp_heap[0] = temp_heap.pop()
                # Réorganiser le tas
                self._heapify_down_custom_min(temp_heap, 0)
            else:
                temp_heap.pop()
        
            return sorted_list

    def _heapify_down_custom(self, heap, index):
     left = 2 * index + 1
     right = 2 * index + 2
     largest = index
    
     if left < len(heap) and heap[left] > heap[largest]:
        largest = left
     if right < len(heap) and heap[right] > heap[largest]:
        largest = right
    
     if largest != index:
        heap[index], heap[largest] = heap[largest], heap[index]
        self._heapify_down_custom(heap, largest)

    def _heapify_down_custom_min(self, heap, index):
     left = 2 * index + 1
     right = 2 * index + 2
     smallest = index
    
     if left < len(heap) and heap[left] < heap[smallest]:
        smallest = left
     if right < len(heap) and heap[right] < heap[smallest]:
        smallest = right
    
     if smallest != index:
        heap[index], heap[smallest] = heap[smallest], heap[index]
        self._heapify_down_custom_min(heap, smallest)

# ==========================================
# === B-ARBRE =============================
# ==========================================

class BTreeNode:
    def __init__(self, order, leaf=False):
        self.order = order
        self.d = (order - 1) // 2  # m = 2*d + 1
        self.keys = []
        self.children = []
        self.leaf = leaf
        self.creation_id = None  # Pour le suivi

class BTree:
    def __init__(self, order):
        if order % 2 == 0:
            raise ValueError("L'ordre m doit être impair (m = 2*d + 1)")
        
        self.order = order
        self.d = (order - 1) // 2
        self.root = BTreeNode(order, True)
        self.creation_counter = 1
    
    def insert(self, key):
      # Cas 1 : arbre vide
      if self.root is None:
        self.root = BTreeNode(self.order, True)
        self.root.keys.append(key)
        self.root.creation_id = self.creation_counter
        self.creation_counter += 1
        return
    
    # Cas 2 : racine pleine → éclatement
      if len(self.root.keys) == (2 * self.d):
        old_root = self.root
        new_root = BTreeNode(self.order, False)
        new_root.children.append(old_root)
        
        # Étape d’éclatement
        self._split_child(new_root, 0, old_root)
        
        # Choisir le bon enfant pour l’insertion
        if key > new_root.keys[0]:
            self._insert_non_full(new_root.children[1], key)
        else:
            self._insert_non_full(new_root.children[0], key)
        
        new_root.creation_id = self.creation_counter
        self.creation_counter += 1
        self.root = new_root
      else:
        # Cas normal
        self._insert_non_full(self.root, key)
    
    def _insert_non_full(self, node, key):
        """Insère dans un nœud qui n'est pas plein"""
        i = len(node.keys) - 1
        
        if node.leaf:
            # Insertion simple dans une feuille
            node.keys.append(key)
            node.keys.sort()
        else:
            # Trouver l'enfant où insérer
            while i >= 0 and node.keys[i] > key:
                i -= 1
            i += 1
            
            # Si l'enfant est plein, le splitter
            if len(node.children[i].keys) == self.order - 1:
                self._split_child(node, i, node.children[i])
                if node.keys[i] < key:
                    i += 1
            
            self._insert_non_full(node.children[i], key)
    
    def _split_child(self, parent, i, child):
      new_child = BTreeNode(self.order, child.leaf)
      d = self.d  # rappel: m = 2*d + 1
    
      # Clé médiane à remonter (kmil)
      median_key = child.keys[d]
    
      # Q reçoit les d plus grandes clés
      new_child.keys = child.keys[d + 1:]
      # P garde les d plus petites
      child.keys = child.keys[:d]
    
      # Répartition des enfants si ce n’est pas une feuille
      if not child.leaf:
        new_child.children = child.children[d + 1:]
        child.children = child.children[:d + 1]
    
      # Insertion de kmil dans le parent
      parent.keys.insert(i, median_key)
      parent.children.insert(i + 1, new_child)
    
      # Marquer la création
      new_child.creation_id = self.creation_counter
      self.creation_counter += 1
      # MÉTHODES DE VISUALISATION ET VALIDATION
    def display_detailed(self):
        """Affiche l'arbre avec le format de votre exemple"""
        if not self.root:
            print("Arbre vide")
            return
        
        print(f"\nB-Adrec d'ordre {self.order}")
        print(f"(Crés. 1 à {self.order-1} par nœud, Enfants: 2 à {self.order})")
        print("-" * 50)
        
        self._display_node_detailed(self.root)
        
        # Afficher le parcours complet pour vérification
        all_keys = self.inorder()
        print(f"\nToutes les clés ({len(all_keys)}): {all_keys}")
    
    def _display_node_detailed(self, node, prefix=""):
        """Affiche un nœud avec le format spécifique"""
        if not node:
            return
        
        # Afficher le nœud courant
        creation_info = f"Céré : {node.creation_id}" if node.creation_id else "Créé : ?"
        children_info = f"{len(node.children)}/{self.order}" if not node.leaf else "0/7"
        
        print(f"{prefix}{node.keys}")
        print(f"{prefix}{creation_info}")
        print(f"{prefix}Enfants : {children_info}")
        print()
        
        # Afficher les enfants
        for i, child in enumerate(node.children):
            self._display_node_detailed(child, prefix + "  ")
    
    def get_all_nodes_info(self):
        """Retourne toutes les informations des nœuds pour l'affichage"""
        nodes_info = []
        self._collect_nodes_info(self.root, nodes_info)
        return nodes_info
    
    def _collect_nodes_info(self, node, nodes_info):
        if node:
            info = {
                'keys': node.keys.copy(),
                'creation_id': node.creation_id,
                'is_leaf': node.leaf,
                'children_count': len(node.children)
            }
            nodes_info.append(info)
            
            for child in node.children:
                self._collect_nodes_info(child, nodes_info)

    # MÉTHODES EXISTANTES (à conserver)
    def search(self, key):
        return self._search(self.root, key)
    
    def _search(self, node, key):
        if not node:
            return False
            
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            return True
        elif node.leaf:
            return False
        else:
            return self._search(node.children[i], key)
    
    def height(self):
        return self._height(self.root)
    
    def _height(self, node):
        if node.leaf:
            return 1
        return 1 + self._height(node.children[0])
    
    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node, result):
        if node:
            for i in range(len(node.keys)):
                if not node.leaf:
                    self._inorder(node.children[i], result)
                result.append(node.keys[i])
            if not node.leaf:
                self._inorder(node.children[len(node.keys)], result)

    def validate_structure(self):
        """Valide que l'arbre respecte TOUTES les règles du cours"""
        errors = self._validate_node(self.root, True, 0, None)
        return errors if errors else []
    
    def _validate_node(self, node, is_root, level, leaf_level):
        """Valide récursivement toutes les propriétés"""
        errors = []
        
        if not node:
            return errors
        
        # Vérifier le nombre de clés
        if is_root:
            # Racine: 1 ≤ k ≤ 2*d
            if len(node.keys) == 0 and len(node.children) > 0:
                errors.append("Racine vide avec enfants")
            if len(node.keys) > (2 * self.d):
                errors.append(f"Racine a trop de clés: {len(node.keys)} > {2 * self.d}")
        else:
            # Nœud non racine: d ≤ k ≤ 2*d
            if len(node.keys) < self.d:
                errors.append(f"Nœud a trop peu de clés: {len(node.keys)} < {self.d}")
            if len(node.keys) > (2 * self.d):
                errors.append(f"Nœud a trop de clés: {len(node.keys)} > {2 * self.d}")
        
        # Vérifier la racine a au moins 2 fils si non feuille
        if is_root and not node.leaf and len(node.children) < 2:
            errors.append("Racine non-feuille doit avoir au moins 2 fils")
        
        # Vérifier l'ordre des clés
        for i in range(len(node.keys) - 1):
            if node.keys[i] >= node.keys[i + 1]:
                errors.append("Clés non triées")
        
        # Vérifier toutes les feuilles au même niveau
        if node.leaf:
            if leaf_level is None:
                leaf_level = level
            elif level != leaf_level:
                errors.append(f"Feuille au niveau {level} ≠ niveau référence {leaf_level}")
        
        # Vérifier récursivement les enfants
        if not node.leaf:
            if len(node.children) != len(node.keys) + 1:
                errors.append("Nombre d'enfants incorrect")
            
            for child in node.children:
                child_errors = self._validate_node(child, False, level + 1, leaf_level)
                errors.extend(child_errors)
        
        return errors
    def average_leaf_depth(self):
        """Calcule la profondeur moyenne des feuilles"""
        depths = []
        self._collect_leaf_depths(self.root, 1, depths)
        return round(sum(depths) / len(depths), 2) if depths else 0

    def _collect_leaf_depths(self, node, level, depths):
        """Récupère récursivement la profondeur de chaque feuille"""
        if node.leaf:
            depths.append(level)
        else:
            for child in node.children:
                self._collect_leaf_depths(child, level + 1, depths)

    def balance_ratio(self):
        """Renvoie le rapport entre profondeur min et max des feuilles"""
        leaf_depths = []
        self._collect_leaf_depths(self.root, 1, leaf_depths)
        if not leaf_depths:
            return 1.0
        min_depth = min(leaf_depths)
        max_depth = max(leaf_depths)
        return round(min_depth / max_depth, 2)

    def balance_quality(self):
        """
        Évalue la qualité de l’équilibre :
        - 1.0 = parfaitement équilibré
        - < 1 = déséquilibre détecté
        """
        ratio = self.balance_ratio()
        if ratio == 1.0:
            return "Excellent (parfaitement équilibré)"
        elif ratio >= 0.8:
            return "Bon équilibre"
        elif ratio >= 0.6:
            return "Moyen"
        else:
            return "Faible équilibre"
# ==========================================
# === FONCTIONS UTILITAIRES ===============
# ==========================================

def generate_abr(values):
    tree = ABR()
    for val in values:
        tree.insert(val)
    return tree

def generate_avl(values):
    tree = AVL()
    for val in values:
        tree.insert(val)
    return tree

def generate_heap(values, heap_type='max'):
    tree = Heap(heap_type)
    for val in values:
        tree.insert(val)
    return tree

def generate_btree(values, order=3):
    tree = BTree(order)
    for val in values:
        tree.insert(val)
    return tree

def abr_properties(tree):
    if tree.root is None:
        return {"Nombre de nœuds": 0, "Hauteur": 0, "État": "Vide"}
    
    nodes = tree.inorder()
    balance = calculate_balance(tree)
    avg_depth = calculate_average_depth(tree)
    
    return {
        "Nombre de nœuds": len(nodes),
        "Hauteur": tree.height(),
        "Profondeur moyenne": avg_depth,
        "Équilibre": f"{balance} (différence hauteur)",
        "Qualité équilibre": calculate_tree_balance_quality(tree),
        "Minimum": min(nodes) if nodes else "-",
        "Maximum": max(nodes) if nodes else "-"
    }

def avl_properties(tree):
    if tree.root is None:
        return {"Nombre de nœuds": 0, "Hauteur": 0, "État": "Vide"}
    
    nodes = tree.inorder()
    balance = tree._get_balance(tree.root) if tree.root else 0
    avg_depth = calculate_average_depth(tree)
    
    return {
        "Nombre de nœuds": len(nodes),
        "Hauteur": tree.height(),
        "Profondeur moyenne": avg_depth,
        "Balance racine": balance,
        "Équilibre": "Parfait" if abs(balance) <= 1 else f"Déséquilibré ({balance})",
        "Qualité équilibre": "Excellent" if abs(balance) <= 1 else "Bon" if abs(balance) <= 2 else "Moyen",
        "Tri (Inorder)": ", ".join(map(str, nodes)) if nodes else "-",
        "Type": "AVL"
    }

def heap_properties(tree):
    if not tree.heap:
        return {"Nombre de nœuds": 0, "Hauteur": 0, "État": "Vide"}
    
    # Calcul du tri
    sorted_values = tree.heap_sort()
    
    n = len(tree.heap)
    if n == 0:
        avg_depth = 0
    else:
        height = tree.height()
        total_depth = sum(min(i.bit_length(), height) for i in range(n))
        avg_depth = round(total_depth / n, 2)
    
    return {
        "Nombre de nœuds": len(tree.heap),
        "Hauteur": tree.height(),
        "Profondeur moyenne": avg_depth,
        "Équilibre": "Structure de tas",
        "Qualité équilibre": "Optimisé pour priorité",
        "Racine": tree.heap[0] if tree.heap else "-",
        "Tri (Heap Sort)": ", ".join(map(str, sorted_values)) if sorted_values else "-",
        "Type": "Tas " + tree.heap_type
    }


def btree_properties(tree):
    nodes = tree.inorder()
    height = tree.height()
    avg_depth = tree.average_leaf_depth()
    balance = tree.balance_ratio()
    balance_quality = tree.balance_quality()
    return {
        "Nombre de nœuds": f"~{len(nodes)} clés au total",
        "Hauteur": height,
        "Ordre (m)": tree.order,
        "Profondeur moyenne": avg_depth,
        "Équilibre": balance,
        "Qualité équilibre": balance_quality,
        "Niveau feuilles": "Tous au même niveau",
        "Parcours trié": ", ".join(map(str, nodes)) if nodes else "Vide",
        "Type": f"B-Arbre académique (m={tree.order}, d={tree.d})"
    }
# Ajoute ces fonctions dans tree_implementations.py

def draw_btree(btree):
    """Dessine un B-Arbre avec SEULEMENT les clés dans les nœuds"""
    G = nx.DiGraph()
    pos = {}
    level_spacing = 1.5
    node_spacing = 2.0
    
    def _add_nodes(node, level=0, index=0, parent=None):
        if node is None:
            return
            
        node_id = f"L{level}_N{index}"
        
        x = index * node_spacing
        y = -level * level_spacing
        pos[node_id] = (x, y)
        
        G.add_node(node_id, 
                  keys=node.keys.copy(),
                  leaf=node.leaf,
                  level=level)
        
        if parent:
            G.add_edge(parent, node_id)
        
        if not node.leaf:
            for i, child in enumerate(node.children):
                child_index = index * len(node.children) + i
                _add_nodes(child, level + 1, child_index, node_id)
    
    _add_nodes(btree.root)
    
    plt.figure(figsize=(14, 10))
    
    # Dessiner les nœuds - SEULEMENT LES CLÉS
    for node_id, node_data in G.nodes(data=True):
        keys = node_data['keys']
        is_leaf = node_data['leaf']
        
        color = "lightgreen" if is_leaf else "lightblue"
        
        # SEULEMENT LES CLÉS ENTRE CROCHETS - PLUS D'INFORMATIONS TECHNIQUES
        node_text = f"[{', '.join(map(str, keys))}]"
        
        plt.gca().text(pos[node_id][0], pos[node_id][1], 
                      node_text,  # ← SEULEMENT LES CLÉS !
                      ha='center', va='center',
                      bbox=dict(boxstyle="round,pad=0.4", 
                               facecolor=color, 
                               edgecolor="black", 
                               alpha=0.8),
                      fontsize=12,  # Police plus grande pour mieux voir
                      fontweight='bold')
    
    # Dessiner les arêtes
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        plt.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                'k-', alpha=0.6, linewidth=1.5)
    
    plt.title(f"B-Arbre d'ordre {btree.order}")
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return image_base64

def draw_tree(root, title="Arbre"):
    """Dessine un arbre binaire classique"""
    if root is None:
        return None
    
    G = nx.DiGraph()
    
    def _add_edges(node, parent=None):
        if node:
            node_id = id(node)
            G.add_node(node_id, key=node.key)
            if parent:
                G.add_edge(parent, node_id)
            _add_edges(node.left, node_id)
            _add_edges(node.right, node_id)
    
    _add_edges(root)
    
    if len(G.nodes) == 0:
        return None
    
    # Positionnement hiérarchique
    pos = _hierarchy_pos(G, id(root))
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, 
            labels={node: data['key'] for node, data in G.nodes(data=True)},
            node_size=1500, node_color="lightblue", 
            font_size=10, font_weight="bold", arrows=False)
    
    plt.title(title)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return image_base64

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Positionnement hiérarchique pour les arbres"""
    pos = {}
    
    def _hierarchy_pos_recursive(G, node, left, right, vert_loc, pos, parent=None):
        pos[node] = ((left + right) / 2, vert_loc)
        neighbors = [n for n in G.neighbors(node) if n != parent]
        
        if len(neighbors) == 2:
            _hierarchy_pos_recursive(G, neighbors[0], left, (left + right) / 2, 
                                   vert_loc - vert_gap, pos, node)
            _hierarchy_pos_recursive(G, neighbors[1], (left + right) / 2, right, 
                                   vert_loc - vert_gap, pos, node)
        elif len(neighbors) == 1:
            _hierarchy_pos_recursive(G, neighbors[0], left, right, 
                                   vert_loc - vert_gap, pos, node)
    
    _hierarchy_pos_recursive(G, root, 0, width, vert_loc, pos)
    return pos
def calculate_balance(tree):
    """Calcule le facteur d'équilibre d'un arbre"""
    if hasattr(tree, 'root') and tree.root:
        if hasattr(tree, '_get_balance'):
            # Pour AVL, utilise la méthode existante
            return abs(tree._get_balance(tree.root))
        else:
            # Pour ABR, calcule la différence de hauteur
            return abs(_height_diff(tree.root))
    return 0

def _height_diff(node):
    """Calcule la différence de hauteur entre sous-arbres gauche et droit"""
    if not node:
        return 0
    left_height = _height(node.left) if node.left else 0
    right_height = _height(node.right) if node.right else 0
    return left_height - right_height

def _height(node):
    """Calcule récursivement la hauteur d'un nœud"""
    if not node:
        return 0
    return 1 + max(_height(node.left), _height(node.right))

def calculate_average_depth(tree):
    """Calcule la profondeur moyenne des nœuds"""
    if hasattr(tree, 'root') and tree.root:
        total_depth = 0
        node_count = 0
        
        def _traverse(node, depth):
            nonlocal total_depth, node_count
            if node:
                total_depth += depth
                node_count += 1
                _traverse(node.left, depth + 1)
                _traverse(node.right, depth + 1)
        
        _traverse(tree.root, 0)
        return round(total_depth / node_count, 2) if node_count > 0 else 0
    return 0

def calculate_tree_balance_quality(tree):
    """Évalue la qualité d'équilibre (0 = parfait, >0 = déséquilibré)"""
    balance_factor = calculate_balance(tree)
    if balance_factor <= 1:
        return "Excellent"
    elif balance_factor <= 2:
        return "Bon"
    elif balance_factor <= 3:
        return "Moyen"
    else:
        return "Faible"
