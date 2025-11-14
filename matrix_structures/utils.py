class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = BSTNode(value)
            return
        self._insert(self.root, value)

    def _insert(self, node, value):
        if value == node.value:
            # on ignore les égaux (ou on peut les placer à droite selon préférence)
            return
        if value < node.value:
            if node.left is None:
                node.left = BSTNode(value)
            else:
                self._insert(node.left, value)
        else:
            if node.right is None:
                node.right = BSTNode(value)
            else:
                self._insert(node.right, value)

    def inorder(self):
        """Retourne la liste triée (parcours infixe)."""
        res = []
        def _in(node):
            if not node:
                return
            _in(node.left)
            res.append(node.value)
            _in(node.right)
        _in(self.root)
        return res

    def to_nested_list(self):
        """Renvoie une structure imbriquée utile pour affichage HTML."""
        def _node_to_tuple(node):
            if node is None:
                return None
            return (node.value, _node_to_tuple(node.left), _node_to_tuple(node.right))
        return _node_to_tuple(self.root)

    def to_html_ul(self):
        """Renvoie un HTML <ul> imbriqué représentant l'arbre."""
        def _html(node):
            if node is None:
                return ''
            left = _html(node.left)
            right = _html(node.right)
            inner = ''
            if left or right:
                inner = f"<ul>{left}{right}</ul>"
            return f"<li>{node.value}{inner}</li>"
        body = _html(self.root)
        if body:
            return f"<ul class=\"bst\">{body}</ul>"
        return '<p>Arbre vide</p>'